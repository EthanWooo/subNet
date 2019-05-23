import sys
from network.strategy_network import StrategyNetwork
import glob

MODEL_SAVE_PATH = './test_models/test_model_subnet.ckpt'
IMAGE_PATH = './images/MyResize'

sn = StrategyNetwork()

dir_path_list = glob.glob(IMAGE_PATH+'/*')
sn.train(dir_path_list, MODEL_SAVE_PATH)
