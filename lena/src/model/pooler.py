import torch.nn as nn
import torch.nn.functional as F


class Pooler(nn.Module):
    def __init__(self):
        super(Pooler, self).__init__()
        
        self.norm = nn.LayerNorm(3 * 4096)
        self.attention = nn.MultiheadAttention(4096, num_heads=32, batch_first=True)
        self.fc1 = nn.Linear(3 * 4096, 2 * 4096)
        self.fc2 = nn.Linear(2 * 4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 4096)
        self.fc5 = nn.Linear(4096, 4096)

        self._initialize_weights()


    def forward(self, vec):
        x1 = self.norm(vec)
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
        x1 = x1.unsqueeze(-1)
        x1 = x1.view(x1.size(0), 3, 4096)
        x1, _ = self.attention(x1, x1, x1)
        x1 = x1.flatten(start_dim=1)
        x1 = F.elu(self.fc1(x1))
        x1 = F.elu(self.fc2(x1))
        x1 = F.elu(self.fc3(x1))
        x1 = F.elu(self.fc4(x1))
        x1 = self.fc5(x1)
        return x1
    
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)