import numpy as np
from numpy import *
class WeightedFusion:

    # 输入的是图像numpy数组
    def fuse(self, image1, image2, image3, saturated_value):
        image1_128 = abs(image1 - 128)
        image2_128 = abs(image2 - 128)
        image3_128 = abs(image3 - 128)
        # image_result = ((image1 * image1_128) + (image2*image2_128))/(image1_128 + image2_128)
        image_result = image2
        image_result[isnan(image_result)] = saturated_value
        return image_result
