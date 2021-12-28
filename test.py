from tokenize import Bracket
import numpy as np
from numpy.core.numeric import outer
import torch
from torch.nn.modules.loss import L1Loss
from autoenc import*
import matplotlib.pyplot as plt





def validation(net, data, label, criterion):
    BATCH_SIZE = 1
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    N = data.shape[0]//BATCH_SIZE 


    id1 = 0
    id2 = BATCH_SIZE

    loss_anomaly = 0
    loss_norm = 0
    for i in range(N):
        net.train()
        with torch.no_grad():
            input = data[id1:id2].to(device)
            out = net(input)

            loss = criterion(out, input)
            if label[i] == 0:
                loss_norm += loss.item()
            elif label[i] == 1:
                loss_anomaly += loss.item()
            id1 += BATCH_SIZE
            id2 += BATCH_SIZE

    print("Anomaly error: ", loss_anomaly/torch.sum(label))
    print("Normal error: ", loss_norm/(len(label) - torch.sum(label)))











