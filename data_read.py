import glob
import pickle

from matplotlib.pyplot import plot
from sound_to_spectrum import*
import numpy as np
from random import shuffle

classes = ["fan"] 
tips = ["train", "test"]

for cls in classes:
	print(cls)
	for tip in tips:
		list_files = glob.glob("/home/liya/research/sound_data/ToyCar/" + tip + "/*.wav")

		shuffle(list_files)

		labels = np.zeros(len(list_files))

		arr = []

		for i, file in enumerate(list_files):
			
			arr_dict = dict()
			
			arr_dict["data"] = detect(to_mel(file))

			if 'anomaly' in file :
				arr_dict["labels"] = 1
			else:
				arr_dict["labels"] = 0

			arr.append(arr_dict)

		with open('data/arr/spectrs' + "_" + cls +"_"+ tip, 'wb') as f:
			pickle.dump(arr, f)
