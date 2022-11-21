
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


def validation(net, data, label, train_type, test_type, cls):
    print("Val")
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    np.random.seed(16)

    y_pred = []
     
    # cnt = 0 
    # cnt_n = 0
    for x in data:
        net.eval()
        with torch.no_grad():
            input = x.float().to(device)
            loss_arr = np.zeros(input.shape[0])
            for i in range(input.shape[0]):

                # cnt += r_pirson(array[n], array[n + 2])
                # cnt_n += 1
                if test_type == "LBL":
                    loss = net.get_lbl(input)
                else:
                    loss = net.get_classic(input)
                    loss_arr[i] = loss.cpu().item()

            y_pred.append(np.mean(loss_arr))

    
    # cnt /= (cnt_n)

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
    # print("Corr: ", cnt)
    # print("Noise acc: ", ls_noise)











