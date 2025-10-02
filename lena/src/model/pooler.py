import torch.nn as nn
import torch.nn.functional as F


class Pooler(nn.Module):
    def __init__(self):
        super(Lena, self).__init__()

        self.projection = nn.Sequential(
            nn.Linear(2 * 4096, 2 * 4096),
            nn.ELU(),
            nn.Linear(2 * 4096, 4096),
            nn.ELU(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ELU(),
            nn.Linear(4096, 4096, bias=False)
        )

        self._initialize_weights()


    def forward(self, x):
        emb = self.projection(x)
        return emb
    
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                gain = nn.init.calculate_gain('leaky_relu', param=1.0)
                nn.init.kaiming_normal_(m.weight, a=1.0, mode='fan_in', nonlinearity='leaky_relu')
                m.weight.data.mul_(gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
