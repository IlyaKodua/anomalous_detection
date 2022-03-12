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

def to_batch_train(file):
    data = load_data(file)
    output = []
    for d in data:
        x = d["data"]
        output.append(torch.from_numpy(x))
    return output



def to_batch_test(file):
    data = load_data(file)
    output = []
    lables  = []
    for d in data:
        # n = d["data"].shape[0]
        output.append(torch.from_numpy(d["data"]))
        lables.append(d["labels"])
    return output, lables

def train(train_type, test_type, cls):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    data = to_batch_train("data/arr/spectrs_" + cls + "_" + "train")

    test_data, label = to_batch_test("data/arr/spectrs_" + cls + "_test")

    net = AE()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # net = forward(net)

    net.to(device)
    criterion.to(device)


    net = nn.DataParallel(net)


    epoch = 1





    for i in range(epoch):
        print("epoch ", i, ", ", epoch)
        for d in data:
            net.train()
            input = d.float().to(device)
            optimizer.zero_grad()
            x1, x2, x3, x5, x6, x7, output = net(input)

            loss = criterion(output, input)
            if train_type == "LBL":
                loss += 0.3*criterion(x1, x7)
                loss += 0.3*criterion(x2, x6)
                loss += 0.3*criterion(x3, x5)

            loss.backward()
            optimizer.step()



        
    validation(net, test_data, label, criterion, train_type, test_type, cls)
        


    torch.save(net.state_dict(), 'net.pth')





