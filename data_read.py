import glob

from numpy.core.fromnumeric import size
from sound_to_spectrum import*
import numpy as np
from random import shuffle


list_files = glob.glob("dev_data_ToyConveyor/ToyConveyor/test/*.wav")

shuffle(list_files)


list_files = list_files[0:1000]
mel = to_mel(list_files[0])

size_len = 5
size_of = [len(list_files)*(mel.shape[1] - size_len), 1,size_len,  mel.shape[0] ]
labels = np.zeros(len(list_files)*(mel.shape[1] - size_len))
arr = np.zeros(size_of)
cnt = 0
for i, file in enumerate(list_files):

  mel = to_mel(file)

  id_start = 0
  id_end = size_len

  for k in range(mel.shape[1] - size_len):
    if 'anomaly' in file:
      labels[cnt] = 1
    arr[cnt,0,:,:] = mel[:, id_start:id_end].T
    id_start += 1
    id_end += 1
    cnt += 1
  print(i)

np.save("val_arr", arr)
np.save("labels_val", labels)