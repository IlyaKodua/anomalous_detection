import scipy.io.wavfile
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

def to_mel(file_name):
  Fs, x = scipy.io.wavfile.read(file_name)
  x = x.astype("float")

  mel = librosa.feature.melspectrogram(y=x, sr=Fs, n_mels=128,
                                    fmax=8000)

  S_dB = librosa.power_to_db(mel, ref=np.max)

  return S_dB





