import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, aliased

from .config_loader import config
from .data_utils.db_models import *
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class PoolerDataset(Dataset):
    def __init__(self, type='train'):

        self.data = self.load_data(type)
         
    def load_data(self, type):
        view1, view2 = [], []
        logging.info('Loading dataset...')
        engine = create_engine(config['db']['url'], echo=False)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        if type == 'train':
            view1_embed = aliased(Train)
            view2_embed = aliased(Train)
            embeddings = torch.load('./lena/data/train_dict.pt')
            
        elif type == 'validation':
            view1_embed = aliased(Validation)
            view2_embed = aliased(Validation)
            embeddings = torch.load('./lena/data/val_dict.pt')

        else:
            raise ValueError("Type must be one of `train`, `validation`.")
        
        records = (session.query(PoolPairs)
                            .filter(PoolPairs.view1_id == view1_embed.function_id)
                            .filter(PoolPairs.view2_id == view2_embed.function_id)
                            .all())
        for poolpair in tqdm(records):
            if poolpair.view1_id in embeddings.keys() and poolpair.view2_id in embeddings.keys():
                view1.append(embeddings[poolpair.view1_id])
                view2.append(embeddings[poolpair.view2_id])
        
        return list(zip(view1,view2))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


'''if __name__ == '__main__':

    PoolerDataset()'''