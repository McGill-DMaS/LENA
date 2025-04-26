import os
import glob
import json

from collections import defaultdict
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from transformers import AutoTokenizer

from ..config_loader import config
from .db_models import *

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

llama = "lena/checkpoints/llama"
tokenizer = AutoTokenizer.from_pretrained(llama, use_fast=True)


def blocks_to_corpus(blocks):
    """
    Convert a Control Flow Graph (CFG) to a formatted corpus string.

    This function processes a list of basic blocks, each representing a segment of assembly instructions
    within the CFG, and concatenates them into a single formatted string. The output is designed for
    feeding into Llama.

    Args:
        blocks (List[dict]): A list of basic blocks comprising the CFG. Each block should encapsulate
                                a sequence of assembly instructions.

    Returns:
        str: A single string containing the formatted assembly code.
    """
    corpuse = []
    for block in blocks:
        text = f"{block['id']}:" + "{ "
        for inst in block['src']:
            if inst[1].startswith("j"):
                calls = f"[{','.join(map(str, block['call']))}]"
                text += f"{inst[1]} {calls}"
            else:
                if inst[1] == 'call':
                    inst[2] = 'func'
                text += " ".join(inst[1:])
            text += " "
        text += "}"
        corpuse.append(text)
    
    src = " , ".join(corpuse)
    src = tokenizer(src)
    src = tokenizer.decode(src['input_ids'][:1000], skip_special_tokens=True)
    return src


def load(mode):
    """
    Load JSON files from the specified directory and convert them into a list of function dictionaries.

    This function reads JSON files corresponding to the specified data mode (train, test, or validation) 
    from a designated directory. The JSON content is expected to be either a list or a dictionary of functions, 
    each of which is validated for the proper format before being included in the returned list.

    Args:
        mode (str): The type of data to load. Must be one of "train", "test", or "validation".

    Returns:
        List[dict]: A list of processed function dictionaries.

    Raises:
        ValueError: If the loaded JSON is neither a list nor a dictionary, or if an individual function 
                    does not adhere to the expected format.
    """
    directory = f'lena/data/{mode}'
    json_files = glob.glob(os.path.join(directory, '*.json'))
    data = []

    if not json_files:
        return data

    for file in json_files:
        with open(file, 'r') as f:
            functions = json.load(f)

    if isinstance(functions, dict):
        functions = [functions]

    if isinstance(functions, list):
        for func in functions:
            if isinstance(func, dict):
                data.append({
                    'name': func['name'],
                    'program': func['program'],
                    'compiler': func['compiler'],
                    'optimization': func['optimization'],
                    'src': blocks_to_corpus(func['blocks'])
                })
            else:
                raise ValueError("Invalid function format.")
    else:
        raise ValueError("Loaded JSON is neither a list nor a dictionary.")

    return data


def write_functions(session, mode, data):
    """
    Insert function records into the 'function' table and create a corresponding record in the train, test,
    or validation table based on the provided mode.

    Args:
        session (Session): The SQLAlchemy session used for database operations.
        mode (str): The type of data to write. Must be one of "train", "test", or "validation".
        data (List[dict]): A list of processed function dictionaries to be inserted.
    """
    for func_data in data:
        # Create a Function instance
        new_function = Function(
            name=func_data["name"],
            program=func_data.get("program"),
            compiler=func_data.get("compiler"),
            optimization=func_data.get("optimization"),
            src=func_data.get("src")
        )
        session.add(new_function)
        session.flush()  # Flush so new_function.id is generated

        # Create a corresponding record in the appropriate table
        if mode == 'train':
            ttv = Train(function_id=new_function.id)
        elif mode == 'test':
            ttv = Test(function_id=new_function.id)
        elif mode == 'validation':
            ttv = Validation(function_id=new_function.id)
        session.add(ttv)
        logging.info(
            f"Inserted function '{new_function.name}' with id {new_function.id} and created a {mode} record."
        )

    session.commit()
    logging.info("All functions and records inserted successfully.")


def write_pool_pairs(session):
    """
    Create PoolPairs records by:
      1. Pairing each non-O0 function with an O0 function of the same compiler.
      2. Chaining O0 functions from different compilers (only one instance per compiler is used).

    Args:
        session (Session): The SQLAlchemy session used for database operations.
    """
    functions = session.query(Function).all()

    # Group functions by (program, name)
    groups = defaultdict(list)
    for func in functions:
        key = (func.program, func.name)
        groups[key].append(func)

    # Process each group for PoolPairs
    for key, funcs in groups.items():
        # Partition functions into O0 and non-O0
        o0_functions = [f for f in funcs if f.optimization == "O0"]
        non_o0_functions = [f for f in funcs if f.optimization != "O0"]

        if not (o0_functions and non_o0_functions):
            continue

        # 1. Pair each non-O0 function with an O0 function of the same compiler
        for f in non_o0_functions:
            matching_o0 = next((o0 for o0 in o0_functions if o0.compiler == f.compiler), None)
            if matching_o0:
                pair = PoolPairs(view1_id=f.id, view2_id=matching_o0.id)
                session.add(pair)
                logging.info(
                    f"PoolPair created: {f.optimization} function id {f.id} (compiler {f.compiler}) "
                    f"paired with O0 id {matching_o0.id}"
                )

        # 2. Chain O0 functions of different compilers (if at least two unique compilers exist)
        if o0_functions:
            # Create a dictionary to keep one O0 function per compiler.
            unique_o0 = {}
            for f in o0_functions:
                # If this compiler isn't seen yet, or we want the one with the lowest id:
                if f.compiler not in unique_o0 or f.id < unique_o0[f.compiler].id:
                    unique_o0[f.compiler] = f

            unique_o0_list = list(unique_o0.values())
            if len(unique_o0_list) > 1:
                # Sort by compiler (or any other criteria)
                o0_sorted = sorted(unique_o0_list, key=lambda f: f.compiler)
                # Chain consecutive functions without looping back.
                for i in range(len(o0_sorted) - 1):
                    pair = PoolPairs(view1_id=o0_sorted[i].id, view2_id=o0_sorted[i+1].id)
                    session.add(pair)
                    logging.info(
                        f"Chained O0 functions: {o0_sorted[i].id} (compiler {o0_sorted[i].compiler}) "
                        f"paired with {o0_sorted[i+1].id} (compiler {o0_sorted[i+1].compiler})"
                    )

    session.commit()
    logging.info("All PoolPairs records inserted successfully.")


if __name__ == '__main__':
    logging.info('Creating/opening database...')
    engine = create_engine(config['db']['url'], echo=False)
    inspector = inspect(engine)

    tables = [
        ('function', Function),
        ('pool_pairs', PoolPairs),
        ('train', Train),
        ('test', Test),
        ('validation', Validation)
    ]

    for table_name, model in tables:
        if not inspector.has_table(table_name):
            model.__table__.create(engine)
            logging.info(f"Table '{table_name}' did not exist and has been created.")

    Base.metadata.create_all(engine)
    logging.info('Database created/opened successfully.')

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        logging.info('Loading train data...')
        data = load('train')
        if len(data) > 0:
            logging.info('Writing train data to db...')
            write_functions(session, 'train', data)

        logging.info('Loading validation data...')
        data = load('validation')
        if len(data) > 0:
            logging.info('Writing validation data to db...')
            write_functions(session, 'validation', data)

        logging.info('Loading test data...')
        data = load('test')
        if len(data) > 0:
            logging.info('Writing test data to db...')
            write_functions(session, 'test', data)

        logging.info('Generating Pooler Pairs...')
        write_pool_pairs(session)

    except Exception as e:
        session.rollback()
        logging.error("Error: " + str(e))
    finally:
        session.close()
