from tokenize import Bracket
import numpy as np
from numpy.core.numeric import outer
import torch
from torch.nn.modules.loss import L1Loss
from autoenc import*
import matplotlib.pyplot as plt
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



def to_batch_train(batch_size):
    data = np.load('train_arr.npy')
    n = data.shape[0]//batch_size
    data = torch.from_numpy(data)
    id_start = 0
    id_end = batch_size
    train = []
    for i in range(n):
        train.append(data[id_start:id_end])
        id_start += batch_size
        id_end += batch_size
    if(id_start != data.shape[0]):
        train.append(data[id_start:data.shape[0]])
    return train


def to_batch_test(batch_size):
    data = np.load('val_arr.npy')
    labels = np.load('labels_val.npy')
    n = data.shape[0]//batch_size
    data = torch.from_numpy(data)
    id_start = 0
    id_end = batch_size
    test = []
    label = []
    for i in range(n):
        test.append(data[id_start:id_end])
        label.append(labels[id_start:id_end])
        id_start += batch_size
        id_end += batch_size
    if(id_start != data.shape[0]):
        test.append(data[id_start:data.shape[0]])
        label.append(labels[id_start:data.shape[0]])
    return test, label

BATCH_SIZE = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = to_batch_train(BATCH_SIZE)
net = AE()
net.apply(initialize_weights)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)


net.to(device)
criterion.to(device)





epoch = 1






for i in range(epoch):
    for k, input in enumerate(data):
        net.train()
        optimizer.zero_grad()
        out = net(input.to(device))

        loss = criterion(out, input)
        loss.backward()
        print(loss.item())
        optimizer.step()


torch.save(net.state_dict(), 'net.pth')





