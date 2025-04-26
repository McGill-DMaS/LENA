import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from transformers import AutoModelForCausalLM, pipeline, BitsAndBytesConfig

import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from ..config_loader import config
from ..lena_dataset import LenaDataset
from ..data_utils.db_models import *
import queue
import numpy as np

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

finetuned_model_dir="lena/checkpoints/llama"
q = queue.Queue()


def collate_fn(batch):

    function_ids, function_masks, keys = np.array(batch, dtype=object).transpose()

    functions = {'input_ids': torch.stack(function_ids.tolist()), 'attention_mask': torch.stack(function_masks.tolist())}


    return functions, keys

def remove_padding(embeddings):

    while embeddings and embeddings[-1] == 0:
        embeddings.pop()

    return embeddings

warnings.filterwarnings("ignore")
            
def main():
    logging.info('Preparing input embeddings (this takes time)...')
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    llama = AutoModelForCausalLM.from_pretrained(
        finetuned_model_dir,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map={"":0}
    )

    llama.eval()
    train_dict, val_dict = {}, {}
    train_set = LenaDataset(type='train')
    val_set = LenaDataset(type='validation')
    batch_size = 24

    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4, prefetch_factor=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4, prefetch_factor=4)

    
    def run_llama(input_ids, attention_mask):
        with torch.inference_mode():
            raw_embeddings = llama(input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True).hidden_states[-1]
            raw_embeddings = raw_embeddings.to(torch.float16)
            masked_embeddings = raw_embeddings * attention_mask.unsqueeze(-1)
            sum_embeddings = masked_embeddings.sum(dim=1)
            count_non_padding = attention_mask.sum(dim=1).unsqueeze(-1)
            mean_embeddings = sum_embeddings / count_non_padding
            squared_diff = (masked_embeddings - mean_embeddings.unsqueeze(1)) ** 2
            variance_embeddings = (squared_diff * attention_mask.unsqueeze(-1)).sum(dim=1) / (count_non_padding  + 1e-6)
            std_embeddings = torch.sqrt(variance_embeddings + 1e-6)

            indices = attention_mask.sum(dim=1) - 1
            LastToken_embedding = raw_embeddings[torch.arange(raw_embeddings.size()[0]),indices,:]
            combined_embeddings = torch.cat((LastToken_embedding, mean_embeddings, std_embeddings), dim=1)
            combined_embeddings = combined_embeddings.detach().cpu()

            return combined_embeddings
    
    
    for i , (data, details) in enumerate(tqdm(train_loader)):
        embeddings = run_llama(data['input_ids'], data['attention_mask'])
        for j in range(embeddings.shape[0]):
            train_dict[details[j]] = embeddings[j]

    logging.info('Saving train embeddings...')
    torch.save(train_dict, './lena/data/train_dict.pt')
    del train_dict

    for i , (data, details) in enumerate(tqdm(val_loader)):
        embeddings = run_llama(data['input_ids'], data['attention_mask'])
        for j in range(embeddings.shape[0]):
            val_dict[details[j]] = embeddings[j]


    logging.info('Saving validation embeddings...')
    torch.save(val_dict, './lena/data/val_dict.pt')



if __name__ == '__main__':

    main()