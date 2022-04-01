import scipy.io.wavfile
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display


def to_mel(file_name, win_size):
	# print(file_name)
	Fs, x = scipy.io.wavfile.read(file_name)
	x = x.astype("float")
	mel = np.abs(librosa.stft(x, n_fft = 2*win_size - 1, hop_length= win_size//2))
	
	# plt.imshow(mel/np.max(mel))

	# plt.show()

	return mel


def norm_to_log(mel):
	# mel = np.log(1 + mel)/ np.log(1 + np.max(mel))
	mel = 2 * (mel - np.min(mel)) / (np.max(mel) - np.min(mel)) - 1
	return mel


def detect(mel, win_size):


	N = len(mel[0,:])
	win = 2

	step = 2*win + 1
	size_array = int((N - 2*win - 1) / step)
	new_array = np.zeros((size_array, 1, 2*win + 1, win_size))

	for i in range(size_array):
		id = win + i*step
		new_array[i,:,:,:] = norm_to_log(mel[:,id-win:id+win+1].T)


	
	return new_array




