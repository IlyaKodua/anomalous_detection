
from tokenize import cookie_re
import torch
from autoenc import*
import matplotlib.pyplot as plt
import numpy as np
from torch.utils import mkldnn as mkldnn_utils
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import cxx_code.code

# def AUC2(loss_norm, loss_anomaly):
#     for
#     return cxx_code.code.AUC(loss_anomaly.tolist(), loss_norm.tolist())*100

def _AUC(loss_norm, loss_anomaly):
    return cxx_code.code.AUC(loss_anomaly.tolist(), loss_norm.tolist())*100

def r_pirson(X,Y):
    cov = torch.sum( (X - torch.mean(X)) * (Y - torch.mean(Y))  )/(torch.prod(torch.tensor(X.shape))-1) 
    cov /= torch.std(X) * torch.std(Y)
    return cov.item()


def validation(net, data, label, criterion, train_type, test_type, cls):
    print("Val")
    BATCH_SIZE = 1024
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    N = data.shape[0]//BATCH_SIZE 

    np.random.seed(16)

    id1 = 0
    id2 = BATCH_SIZE

    loss_anomaly = []
    loss_norm = []
     
    cnt = 0 
    for i in range(N):
        net.eval()
        with torch.no_grad():
            input = data[id1:id2].to(device)
            batch_lab = label[id1:id2].to(device)
            x1, x2, x4, x5, x6 = net(input)
            cnt += r_pirson(x2, x4)
            for i in range(input.shape[0]):
                loss = criterion(x6[i], input[i])
                if(test_type == "LBL"):
                    loss += criterion(x1[i], x5[i])
                    loss += criterion(x2[i], x4[i])
                if batch_lab[i] == 0:
                    loss_norm.append(loss.item())
                elif batch_lab[i] == 1:
                    loss_anomaly.append(loss.item())
            id1 += BATCH_SIZE
            id2 += BATCH_SIZE
    

    id2 = data.shape[0]
    with torch.no_grad():
        input = data[id1:id2].to(device)
        batch_lab = label[id1:id2].to(device)
        x1, x2, x4, x5, x6 = net(input)
        cnt += r_pirson(x2, x4)
        for i in range(input.shape[0]):
            loss = criterion(x6[i], input[i])
            if(test_type == "LBL"):
                loss += criterion(x1[i], x5[i])
                loss += criterion(x2[i], x4[i])
            if batch_lab[i] == 0:
                loss_norm.append(loss.item())
            elif batch_lab[i] == 1:
                loss_anomaly.append(loss.item())
    cnt /= (N+1)
    # loss_noise = []
    # with torch.no_grad():
    #     input = torch.rand((128,1,5,128)).to(device)
    #     x1, x2, x4, x5, x6 = net(input)

    #     for i in range(128):
    #         loss = criterion(x6[i], input[i])
    #         loss += criterion(x1[i], x5[i])
    #         loss += criterion(x2[i], x4[i])
    #     loss_noise.append(loss.item())


    loss_anomaly = np.array(loss_anomaly)
    np.random.shuffle(loss_anomaly)
    # loss_anomaly.reshape((1, len(loss_anomaly)))

    loss_norm = np.array(loss_norm)
    np.random.shuffle(loss_norm)
    # loss_norm = loss_norm.reshape((len(loss_norm), 1))


    # loss_noise = np.array(loss_noise)
    # np.random.shuffle(loss_noise)
    # loss_noise = loss_noise.reshape((1, len(loss_noise)))

    # loss_noise = np.array(loss_noise)
    ls_anom = _AUC(loss_norm.astype("float32"), loss_anomaly.astype("float32"))
    print("class: ", cls)
    print("test type: ", test_type, ", train type: ", train_type)
    print("Anomaly acc: ", ls_anom, " %")
    print("Corr: ", cnt)
    # print("Noise acc: ", ls_noise)











