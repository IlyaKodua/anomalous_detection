import torch
import torch.nn as nn



class AE(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(1),
            nn.Linear(128, 128),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(1),
            nn.Linear(128,8),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(1),
            nn.Linear(8,128),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(1),
            nn.ReLU6(inplace=True),
            nn.Linear(128,128),
            nn.BatchNorm2d(1),
            nn.ReLU6(inplace=True),
            nn.Linear(128,128),
        )

    def forward(self,x):
        return self.model(x.float())