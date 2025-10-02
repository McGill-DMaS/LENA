import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from ..pooler_dataset import PoolerDataset
from torch.utils.data import DataLoader
from ..config_loader import config
from ..model.pooler import Pooler
import math
import numpy as np
from torch.amp import autocast, GradScaler
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def collate_fn(batch):
    view1 = torch.stack([item[0] for item in batch])
    view2 = torch.stack([item[1] for item in batch])
    return torch.nan_to_num(view1, nan=0.0, posinf=0.0, neginf=0.0), torch.nan_to_num(view2, nan=0.0, posinf=0.0, neginf=0.0)

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def lr_lambda(step):
    warmup_steps = 10
    if step < warmup_steps:
        return step / warmup_steps
    return 0.5 * (1 + torch.cos((step - warmup_steps) / ((3169*15/4) - warmup_steps) * torch.tensor(3.141592653589793)))

def main():
    device = "cuda:0"
    scaler = GradScaler()

    lr = 2e-5
    std_coeff = 5.0
    cov_coeff = 1.0
    alpha = 0.6
    
    epochs = 50
    report_steps = 20
  
    dataset = PoolerDataset(type='train')
    valid_dataset = PoolerDataset(type='validation')

    model = Pooler()
    model.to(device)
    
    batch_size = 1024
    
    total = len(dataset)

    def info_nce_loss(x: torch.Tensor,
                  y: torch.Tensor,
                  temperature: float = 0.07) -> torch.Tensor:

        x = F.normalize(x, p=2, dim=1)   
        y = F.normalize(y, p=2, dim=1)  

        logits = x @ y.t() 
        logits = logits / temperature

        batch_size = x.shape[0]
        targets = torch.arange(batch_size, device=x.device)

        loss_xy = F.cross_entropy(logits, targets)
        loss_yx = F.cross_entropy(logits.t(), targets)

        return 0.5 * (loss_xy + loss_yx)

    def VICRegLoss(x, y, std_coeff, num_features = 4096):

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(num_features)

        loss = (std_coeff * std_loss
            + cov_coeff * cov_loss
        )
        return loss, std_loss, cov_loss
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    model.train()
    best_val_loss = math.inf
    logging.info('Running dataloader...')
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=3, prefetch_factor=2)
    validation_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=3, prefetch_factor=2)
    logging.info('Dataloader is ready.')

    scheduler = CosineAnnealingLR(
        optimizer,
        eta_min=1e-10,
        T_max=epochs * len(dataloader)
    )

    for epoch in range(epochs):
        
        std_coeff += 0.10
        
        for i , (view1, view2) in enumerate(dataloader):
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                z1 = model(view1.to(device))
                z2 = model(view2.to(device))

                loss = VICRegLoss(z1, z2, std_coeff, num_features = 4096)

            scaler.scale(loss).backward() 
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if i%report_steps == 0:
                logging.info("Running validation...")

                valid_loss = []
                for j , (view1, view2) in enumerate(validation_dataloader):
                    with torch.no_grad():
                        with autocast(device_type='cuda'):
                            z1 = model(view1.to(device))
                            z2 = model(view2.to(device))
                            validation_loss = VICRegLoss(z1, z2, 25, num_features = 4096)
                        
                        valid_loss.append(validation_loss.item())

                avg_loss = np.mean(valid_loss)

                if avg_loss <= best_val_loss:
                    save_path = 'lena/checkpoints/pooler/'

                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    torch.save(model.state_dict(), f'{save_path}model.pth')
                    best_val_loss = avg_loss
  
                logging.info(f"Epoch [{epoch+((i)/(math.ceil(total/batch_size))):.4f}], Loss: {loss.item():.4f}, Best_val_loss: {best_val_loss:.2f}")
               
    
        


if __name__ == '__main__':
    main()
