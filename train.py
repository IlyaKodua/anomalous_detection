import numpy as np
from scipy.ndimage.measurements import label
import torch
from torch.nn.modules.loss import L1Loss
from autoenc import*
import matplotlib.pyplot as plt
import pickle
from test import*
BATCH_SIZE = 1024


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



def load_data(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def get_size(array):
    batch = 0
    for a in array:
        batch += a.shape[0]
    size2 = array[0][0].shape
    return (batch, size2[0], size2[1], size2[2])

def to_batch_train():
    data = load_data('arrtrain')
    size = get_size(data)
    data_array = torch.zeros(size)
    cnt = 0
    for array in data:
        for a in array:
            data_array[cnt,:,:,:] = torch.from_numpy(a)
            cnt += 1
    return data_array



def to_batch_test():
    data = load_data('arrtest')
    labels = load_data('labelstest')
    size = get_size(data)
    label = torch.zeros(size[0])
    data_array = torch.zeros(size)
    cnt = 0
    for i,array in enumerate(data):
        for a in array:
            data_array[cnt,:,:,:] = torch.from_numpy(a)
            if labels[i] == 1:
                label[cnt] = 1
            cnt += 1
    return data_array, label

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



data = to_batch_train()

test_data, label = to_batch_test()

net = AE()
net.apply(initialize_weights)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# net = forward(net)

net.to(device)
criterion.to(device)


net = nn.DataParallel(net)


epoch = 100



N = data.shape[0]//BATCH_SIZE 

epoch_cnt = 0
for i in range(epoch):
    id1 = 0
    id2 = BATCH_SIZE
    for i in range(N):
        net.train()
        input = data[id1:id2].to(device)
        optimizer.zero_grad()
        x1, x2, x4, x5, x6 = net(input)

        loss = criterion(x6, input)
        # loss += criterion(x1, x5)
        # loss += criterion(x2, x4)

        loss.backward()
        optimizer.step()
        id1 += BATCH_SIZE
        id2 += BATCH_SIZE
    epoch_cnt += 1
    print(epoch_cnt)


    
validation(net, test_data, label, criterion)
    


torch.save(net.state_dict(), 'net.pth')





