import glob
import numpy as np
from util.ImageProcessor import ImageProcessor

image_processor = ImageProcessor()
IMAGE_PATH = 'images/MyResize'


dir_path_list = glob.glob(IMAGE_PATH+'/*')

image_processor.generate_training_data(dir_path_list)