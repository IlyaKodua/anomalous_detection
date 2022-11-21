from sklearn import metrics
import numpy as np
from train import*
import time

classes = ["valve"]


test_types = ["classic"]
train_types = ["LBL"]

for cls in classes:
    for train_type in train_types:
        for test_type in test_types:
            start = time.time() 
            train(train_type, test_type, cls, 8)
