import glob
import pickle

from matplotlib.pyplot import plot
from sound_to_spectrum import*
import numpy as np
from random import shuffle

classes = ["pump"] 
tips = ["train", "test"]
win_size = 8

for cls in classes:
	print(cls)
	for tip in tips:
		list_files = glob.glob("/home/liya/research/sound_data/" + cls + "/" + tip + "/*.wav")

		shuffle(list_files)

		labels = np.zeros(len(list_files))

		arr = []

		for i, file in enumerate(list_files):
			
			print(int((i + 1) / len(list_files) * 100), " %")
			arr_dict = dict()
			
			arr_dict["data"] = detect(to_mel(file, win_size), win_size)

			if 'anomaly' in file :
				arr_dict["labels"] = 1
			else:
				arr_dict["labels"] = 0

			arr.append(arr_dict)

		with open('data/arr/spectrs' + "_" + cls +"_"+ tip, 'wb') as f:
			pickle.dump(arr, f)
