from sklearn import metrics
import numpy as np
from train import*
import time

classes = ["fan", "pump", "slider", "ToyCar", "ToyConveyor", "valve" ]


test_types = ["classic", "LBL"]
train_types = ["classic", "LBL"]

for cls in classes:
    for train_type in train_types:
        for test_type in test_types:
            start = time.time() 
            train(train_type, test_type, cls)
