import scipy.io.wavfile
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display


def to_mel(file_name):
	# print(file_name)
	Fs, x = scipy.io.wavfile.read(file_name)
	x = x.astype("float")
	mel = librosa.feature.melspectrogram(y=x, sr=Fs, n_mels=128,
                                    win_length=1024, hop_length=512)
	
	return mel


def norm_to_log(mel):
	# mel = np.log(1 + mel)/ np.log(1 + np.max(mel))
	mel = 2 * (mel - np.min(mel)) / (np.max(mel) - np.min(mel)) - 1
	return mel


def detect(mel):


	N = len(mel[0,:])
	win = 2

	step = 2*win + 1
	size_array = int((N - 2*win - 1) / step)
	new_array = np.zeros((size_array, 1, 2*win + 1, 128))

	for i in range(size_array):
		id = win + i*step
		new_array[i,:,:,:] = norm_to_log(mel[:,id-win:id+win+1].T)


	
	return new_array




