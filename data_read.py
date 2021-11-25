import glob
from sound_to_spectrum import*
import numpy as np


N = 200
Fs = 30
x = np.arange(N)/N


y = np.log(2 + np.sin(2*np.pi*30*x))

spc = np.abs(np.fft.fft(y))
plt.plot(spc)
plt.show()

# list_files = glob.glob("/media/ilya/TOSHIBA EXT/dev_data_ToyConveyor/ToyConveyor/train/*.wav")


# mel = to_mel(list_files[0])

# size_of = [len(list_files), 1, mel.shape[0], mel.shape[1]]
# arr = np.zeros(size_of)

# for i, file in enumerate(list_files):
#   arr[i,0,:,:] = to_mel(file)
#   print(i)

# np.save("arr", arr)