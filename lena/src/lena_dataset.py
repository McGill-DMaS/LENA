import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from transformers import AutoTokenizer

from .config_loader import config
from .data_utils.db_models import *
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

llama = "lena/checkpoints/llama"
tokenizer = AutoTokenizer.from_pretrained(llama, use_fast=True)

class LenaDataset(Dataset):
    def __init__(self, type='train'):

        self.data = self.load_data(type)
        

    def createPrompt(self, func):
    
        bos_token = '<s>'
        system_prompt = f"[INST] The O0 form of \n"
        input_prompt = f" {func} is [/INST]" 
        
        return bos_token + system_prompt + input_prompt
    
    def load_data(self, type):
        tokens, detail = [], []
        logging.info('Loading dataset...')
        engine = create_engine(config['db']['url'], echo=False)
        Session = sessionmaker(bind=engine)
        session = Session()
        if type == 'train':
            records = session.query(Train).all()
        elif type == 'validation':
            records = session.query(Validation).all()
        elif type == 'test':
            records = session.query(Test).all()
        else:
            raise ValueError("Type must be one of `train`, `validation`, or `test`.")
        
        for record in records:
            function = record.function
            prompt = self.createPrompt(function.src)
            tokens.append(prompt)
            if type == 'test':
                detail.append([function.id, function.name, function.program, function.compiler, function.optimization])
            else:
                detail.append(function.id)
        
        logging.info('Tokenizing...')

        tokens = tokenizer(tokens, return_tensors="pt", max_length=config['llama']['input_length'], truncation=True, padding='max_length')

        return list(zip(tokens['input_ids'],tokens['attention_mask'] , detail))
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
