
from tokenize import cookie_re
import torch
from autoenc import*
import matplotlib.pyplot as plt
import numpy as np
from torch.utils import mkldnn as mkldnn_utils
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn import metrics
# def AUC2(loss_norm, loss_anomaly):
#     for
#     return cxx_code.code.AUC(loss_anomaly.tolist(), loss_norm.tolist())*100


def r_pirson(X,Y):
    cov = torch.sum( (X - torch.mean(X)) * (Y - torch.mean(Y))  )/(torch.prod(torch.tensor(X.shape))-1) 
    cov /= torch.std(X) * torch.std(Y)
    return cov.item()

def MSE_loss(x, y):
    a = torch.mean((x - y)**2, dim=(1,2,3))
    return a

def validation(net, data, label, criterion, train_type, test_type, cls):
    print("Val")
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    np.random.seed(16)

    y_pred = []
     

    for i, x in enumerate(data):
        net.eval()
        print(int((i+1)/len(data)*100), " %")
        with torch.no_grad():
            input = x.float().to(device)
            output = net(input)

            loss = MSE_loss(output, input)
            
            y_pred.append(np.mean(loss.cpu().numpy()))

    

    y_pred = np.array(y_pred)
    y_true = np.array(label)

    # loss_noise = np.array(loss_noise)
    auc = metrics.roc_auc_score(y_true, y_pred)
    pauc = metrics.roc_auc_score(y_true, y_pred, max_fpr = 0.1)
    fpr, tpr, thresholds = metrics.roc_curve(1 + y_true, y_pred, pos_label=2)
    auc2 = metrics.auc(fpr, tpr)

    print("class: ", cls)
    print("test type: ", test_type, ", train type: ", train_type)
    print("Anomaly acc: ", auc, " %")
    print("Anomaly acc 2: ", auc2, " %")
    print("pAnomaly acc: ", pauc, " %")
    # print("Noise acc: ", ls_noise)











