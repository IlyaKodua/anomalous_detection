from tokenize import Bracket
import numpy as np
from numpy.core.numeric import outer
import torch
from torch.nn.modules.loss import L1Loss
from autoenc import*

def initialize_weights(m):
  if isinstance(m, nn.Conv2d):
      nn.init.xavier_normal_(m.weight.data)
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
      nn.init.xavier_uniform(m.weight.data)
      nn.init.constant_(m.bias.data, 0)



def to_batch(batch_size, device):
    data = np.load('train_arr.npy')
    n = data.shape[0]//batch_size
    data = torch.from_numpy(data).to(device)
    id_start = 0
    id_end = batch_size
    train = []
    for i in range(n):
        train.append(data[id_start:id_end])
        id_start += batch_size
        id_end += batch_size
    train.append(data[id_start:data.shape[0]])
    return train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = to_batch(32, device)
net = AE()
net.apply(initialize_weights)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)


net.to(device)
criterion.to(device)


rd = torch.rand(1,1,5,128).to(device)


epoch = 1


for i in range(epoch):
    for k, input in enumerate(data):
        if(k == 1000):
            break
        net.train()
        optimizer.zero_grad()
        out = net(input)

        loss = criterion(out, input)
        loss.backward()
        print(loss.item())
        optimizer.step()
    loss = criterion(rd, net(rd))
    print(loss.item())



