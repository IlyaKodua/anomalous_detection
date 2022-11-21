import torch
import torch.nn as nn
import torch.nn.functional as F



class AE4(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.model = nn.ModuleList()

        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1)))
        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1)))
        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1)))
        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.Tanh()))

    def get_classic(self, x1):
        x1, x2, x3, x4, x5  = self.forward(x1)
        return F.mse_loss(x1,x5)

    def get_lbl(self, x1):
        x1, x2, x3, x4, x5  = self.forward(x1)
        return F.mse_loss(x1,x5) + F.mse_loss(x2,x4)

    def forward(self,x1):
        x2 = self.model[0](x1)
        x3 = self.model[1](x2)
        x4 = self.model[2](x3)
        x5 = self.model[3](x4)
        return x1, x2, x3, x4, x5


class AE5(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.model = nn.ModuleList()

        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1)))
        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1)))
        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1)))
        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1)))
        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.Tanh()))

    def get_classic(self, x1):
        x1, x2, x3, x4, x5, x6  = self.forward(x1)
        return F.mse_loss(x1,x6)

    def get_lbl(self, x1):
        x1, x2, x3, x4, x5, x6  = self.forward(x1)
        return F.mse_loss(x1,x6) + F.mse_loss(x2,x5) + F.mse_loss(x3,x4)




    def forward(self,x1):
        x2 = self.model[0](x1)
        x3 = self.model[1](x2)
        x4 = self.model[2](x3)
        x5 = self.model[3](x4)
        x6 = self.model[4](x5)
        return x1, x2, x3, x4, x5, x6 


class AE6(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.model = nn.ModuleList()

        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1)))
        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1)))
        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(5)))
        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(5)))
        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(5)))
        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.Tanh()))

    def get_classic(self, x1):
        x1, x2, x3, x4, x5, x6, x7  = self.forward(x1)
        return F.mse_loss(x1,x7)

    def get_lbl(self, x1):
        x1, x2, x3, x4, x5, x6, x7  = self.forward(x1)
        return F.mse_loss(x1,x7) + F.mse_loss(x2,x6) + F.mse_loss(x3,x5)




    def forward(self,x1):
        x2 = self.model[0](x1)
        x3 = self.model[1](x2)
        x4 = self.model[2](x3)
        x5 = self.model[3](x4)
        x6 = self.model[4](x5)
        x7 = self.model[5](x6)
        return x1, x2, x3, x4, x5, x6, x7 



class AE7(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.model = nn.ModuleList()

        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True)))
        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True)))
        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True)))
        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True)))
        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True)))
        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True)))
        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True)))
        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.Tanh()))

    def get_classic(self, x1):
        x1, x2, x3, x4, x5, x6, x7, x8, x9  = self.forward(x1)
        return F.mse_loss(x1,x9)

    def get_lbl(self, x1):
        x1, x2, x3, x4, x5, x6, x7, x8, x9  = self.forward(x1)
        return F.mse_loss(x1,x9) + F.mse_loss(x2,x8) + F.mse_loss(x3,x7)+ F.mse_loss(x4,x6)

    def forward(self,x1):
        x2 = self.model[0](x1)
        x3 = self.model[1](x2)
        x4 = self.model[2](x3)
        x5 = self.model[3](x4)
        x6 = self.model[4](x5)
        x7 = self.model[5](x6)
        x8 = self.model[6](x7)
        x9 = self.model[7](x8)
        return x1, x2, x3, x4, x5, x6, x7, x8, x9




class AE8(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.model = nn.ModuleList()

        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True)))
        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True)))
        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True)))
        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True)))
        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True)))
        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True)))
        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True)))
        self.model.append(nn.Sequential(
            nn.Linear(128,128),
            nn.Tanh()))

    def get_classic(self, x1):
        x1, x2, x3, x4, x5, x6, x7, x8, x9  = self.forward(x1)
        return F.mse_loss(x1,x9)

    def get_lbl(self, x1):
        x1, x2, x3, x4, x5, x6, x7, x8, x9  = self.forward(x1)
        return F.mse_loss(x1,x9) + F.mse_loss(x2,x8) + F.mse_loss(x3,x7)+ F.mse_loss(x4,x6)

    def forward(self,x1):
        x2 = self.model[0](x1)
        x3 = self.model[1](x2)
        x4 = self.model[2](x3)
        x5 = self.model[3](x4)
        x6 = self.model[4](x5)
        x7 = self.model[5](x6)
        x8 = self.model[6](x7)
        x9 = self.model[7](x8)
        return x1, x2, x3, x4, x5, x6, x7, x8, x9