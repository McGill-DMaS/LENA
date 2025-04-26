import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

from datasets import load_dataset
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from ..config_loader import config
from ..lena_dataset import LenaDataset
from ..data_utils.db_models import *
from ..model.pooler import Pooler
import queue
import numpy as np
import threading
import ast
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
    logging.info('Generating embeddings (this takes time)...')
    device_id = 0
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    llama = AutoModelForCausalLM.from_pretrained(
        finetuned_model_dir,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map={"":device_id}
    )

    llama.eval()
    pooler = Pooler()
    pooler = pooler.half()
    pooler.load_state_dict(torch.load('lena/checkpoints/pooler/model.pth',
        map_location=torch.device('cuda:0')), strict=False)
    llama.eval()
    pooler.to(f'cuda:{device_id}')
    pooler.eval()
    inference_dict = {}
    test_set = LenaDataset(type='test')
    batch_size = 24

    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4, prefetch_factor=4)
    
    
    for i , (data, details) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            torch.cuda.synchronize()
            raw_embeddings = llama(data['input_ids'], attention_mask=data['attention_mask'], output_hidden_states=True, return_dict=True).hidden_states[-1]
            raw_embeddings = raw_embeddings.to(torch.float16)
            masked_embeddings = raw_embeddings * data['attention_mask'].unsqueeze(-1)
            sum_embeddings = masked_embeddings.sum(dim=1)
            count_non_padding = data['attention_mask'].sum(dim=1).unsqueeze(-1)
            mean_embeddings = sum_embeddings / count_non_padding
            squared_diff = (masked_embeddings - mean_embeddings.unsqueeze(1)) ** 2
            variance_embeddings = (squared_diff * data['attention_mask'].unsqueeze(-1)).sum(dim=1) / (count_non_padding  + 1e-6)
            std_embeddings = torch.sqrt(variance_embeddings + 1e-6)

            indices = data['attention_mask'].sum(dim=1) - 1
            LastToken_embedding = raw_embeddings[torch.arange(raw_embeddings.size()[0]),indices,:]
            combined_embeddings = torch.nan_to_num(torch.cat((LastToken_embedding, mean_embeddings, std_embeddings), dim=1).squeeze().half(), nan=0.0, posinf=0.0, neginf=0.0)
            embeddings = pooler(combined_embeddings.to(f'cuda:{device_id}'))

        for j in range(embeddings.shape[0]):
            inference_dict[details[j][0]] = {'name':details[j][1],'program':details[j][2],'compiler':details[j][3], 'optimization':details[j][4], 'embedding': embeddings[j].cpu()}

    logging.info('Saving inference embeddings...')
    torch.save(inference_dict, './lena/data/inference_dict.pt')




if __name__ == '__main__':

    main()