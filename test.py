from tokenize import Bracket
import numpy as np
from numpy.core.numeric import outer
import torch
from torch.nn.modules.loss import L1Loss
from autoenc import*
import matplotlib.pyplot as plt





def to_batch_test(batch_size, device):
    data = np.load('val_arr.npy')
    labels = np.load('labels_val.npy')
    n = data.shape[0]//batch_size
    data = torch.from_numpy(data).to(device)
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

net = AE()

net.load_state_dict(torch.load('net.pth'))
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)


net.to(device)
criterion.to(device)





loss_an = []
loss_norm = []

test_data, label = to_batch_test(1, device)

for k, input in enumerate(test_data):
    net.eval()
    out = net(input)
    loss = criterion(out, input)
    if(len(loss_an) > 10000):
        break
    if label[k] == 1:
        loss_an.append(loss.item())
    else:
        loss_norm.append(loss.item())
    print(len(loss_an))


plt.figure(1)
plt.hist(loss_an, 100)



plt.figure(2)
plt.hist(loss_norm, 100)

plt.show()









