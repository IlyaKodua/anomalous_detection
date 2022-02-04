
import torch
from autoenc import*
import matplotlib.pyplot as plt
import numpy as np




def validation(net, data, label, criterion):
    BATCH_SIZE = 1
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    N = data.shape[0]//BATCH_SIZE 


    id1 = 0
    id2 = BATCH_SIZE

    loss_anomaly = []
    loss_norm = []

    
    for i in range(N):
        net.train()
        with torch.no_grad():
            input = data[id1:id2].to(device)
            out = net(input)

            loss = criterion(out, input)



            if label[i] == 0:
                loss_norm.append(loss.item())
            elif label[i] == 1:
                loss_anomaly.append(loss.item())
            id1 += BATCH_SIZE
            id2 += BATCH_SIZE

    loss_anomaly = np.array(loss_anomaly)
    loss_norm = np.array(loss_norm)
    # ls_a = loss_anomaly[loss_anomaly > 1.5*np.mean(loss_anomaly)]
    # ls_n = loss_norm[loss_norm > 1.5*np.mean(loss_norm)]
    ls_a =  np.mean(loss_anomaly)
    ls_n = np.mean(loss_norm)
    print("Anomaly error: ", ls_a)
    print("Normal error: ", ls_n)
    print(int((ls_a - ls_n)/ls_n*100), ' %')
    return loss_anomaly, loss_norm






