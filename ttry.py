import scipy.io as sio
import numpy as np

flow1 = sio.loadmat('./images/MyResize/01/h2r256channal.mat')
flow1 = flow1['future']
flow1 = flow1[np.newaxis, :, :, :]
print(flow1.shape)