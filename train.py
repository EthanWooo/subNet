import sys
sys.path.append('/home/wys/project/subnet')

from network.DeepFuseNetwork import DeepFuseNetwork
import glob

# './models/test_model.ckpt'
MODEL_SAVE_PATH = './models'
LEARNING_RATE = 1e-3
IMAGE_PATH = './images/MyResize'

dfn = DeepFuseNetwork()

dir_path_list = glob.glob(IMAGE_PATH+'/*')
dfn.train(dir_path_list, MODEL_SAVE_PATH, epoches=1000, batch_size=1, learning_rate=LEARNING_RATE)
