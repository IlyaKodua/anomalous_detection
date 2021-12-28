import scipy.io.wavfile
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display


def to_mel(file_name):
	print(file_name)
	Fs, x = scipy.io.wavfile.read(file_name)
	x = x.astype("float")
	mel = librosa.feature.melspectrogram(y=x, sr=Fs, n_mels=128,
                                    win_length=1024, hop_length=512)
	
	return mel


def norm_to_log(mel):
	mel -= np.min(mel)
	mel *= (np.exp(1) - np.exp(-1))/(np.max(mel) + 1e-3)
	mel += np.exp(-1)
	return np.log(mel)


def detect(mel):

	mean_energi = np.mean(mel) * 128

	detected_array = set()

	N = len(mel[0,:])
	win = 2
	for ti in range(N):
		if np.sum(mel[:,ti]) > mean_energi*1.5 and ti >= win and ti < N - win :
			detected_array.add(ti)

	plt.show()
	new_array = np.zeros((len(detected_array), 1, 5, 128))

	cnt = 0
	for ti in detected_array:
		new_array[cnt,:,:,:] = norm_to_log(mel[:,ti-win:ti+win+1].T)
		cnt += 1
	
	return new_array




