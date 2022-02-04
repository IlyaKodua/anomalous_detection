import glob
import pickle

from matplotlib.pyplot import plot
from sound_to_spectrum import*
import numpy as np
from random import shuffle


tips = ["train", "test"]


for tip in tips:
	list_files = glob.glob("/home/liya/research/sound_data/ToyCar/" + tip + "/*.wav")

	shuffle(list_files)

	labels = np.zeros(len(list_files))

	arr = []

	for i, file in enumerate(list_files):
		
		if 'anomaly' in file :
			labels[i] = 1
		
		arr.append(detect(to_mel(file)))
		# plt.imshow(arr[0][0][0])
		# plt.show()
		print('Calculating: ', int(i/len(list_files)*100), ' %')


	with open('arr' + tip, 'wb') as f:
		pickle.dump(arr, f)


	if(tip == "test"):
		with open('labels' + tip, 'wb') as f:
			pickle.dump(labels, f)
