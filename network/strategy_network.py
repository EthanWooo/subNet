import tensorflow as tf
import scipy
from network.base.ConvolutionLayer import ConvLayer
import tensorflow as tf
from util.ImageProcessor import ImageProcessor
import scipy.io as scio
import random
import numpy as np
import os
import scipy.io as sio

# MODEL_SAVE_PATH = './test_model/test_model_subnet.ckpt'


class StrategyNetwork:
    __kernel_number0 = 1280
    __kernel_number1 = 1280
    # 第一层卷积核大小是7 x 7，对输入的96*144*1280进行卷积, 提取1280个特征
    __shape_kernel_conv1 = [7, 7, __kernel_number0, __kernel_number1]
    # 第一层卷积层步长
    __strides1 = 2
    __shape_strides1 = [1, __strides1, __strides1, 1]
    # 第一层卷积层padding方式
    __padding1 = 'SAME'
    __activation_func1 = tf.nn.relu
    # scope名称
    __scope_conv1 = 'concat_conv1'
    __output_shape1 = []

    __kernel_number2 = 1024
    # 第二层卷积核大小是3 x 3，对生成的96*144*1280个特征进行卷积，提取1024个特征
    __shape_kernel_conv2 = [3, 3, __kernel_number1, __kernel_number2]
    # 第二层卷积层步长
    __strides2 = 2
    __shape_strides2 = [1, __strides2, __strides2, 1]
    # 第二层卷积层padding方式
    __padding2 = 'SAME'
    __activation_func2 = tf.nn.relu
    # scope名词
    __scope_conv2 = 'concat_conv2'

    __kernel_number3 = 1024
    # 第三层卷积核大小是3 x 3，对生成的48*72*1024个特征进行卷积，提取2048个特征
    __shape_kernel_conv3 = [3, 3, __kernel_number2, __kernel_number3]
    # 第三层卷积层步长
    __strides3 = 2
    __shape_strides3 = [1, __strides3, __strides3, 1]
    # 第三层卷积层padding方式
    __padding3 = 'SAME'
    __activation_func3 = tf.nn.relu
    # scope名词
    __scope_conv3 = 'concat_conv3'

    # __kernel_number4 = 1024
    # # 第三层卷积核大小是3 x 3，对生成的2048个特征进行卷积，提取1024个特征
    # __shape_kernel_conv4 = [3, 3, __kernel_number3, __kernel_number4]
    # # 第三层卷积层步长
    # __strides4 = 1
    # __shape_strides4 = [1, __strides4, __strides4, 1]
    # # 第三层卷积层padding方式
    # __padding4 = 'SAME'
    # __activation_func4 = tf.nn.relu
    # # scope名词
    # __scope_conv4 = 'concat_conv4'
    #
    # __kernel_number5 = 2048
    # # 第三层卷积核大小是5 x 5，对生成的2048个特征进行卷积，提取2048个特征
    # __shape_kernel_conv5 = [5, 5, __kernel_number4, __kernel_number5]
    # # 第三层卷积层步长
    # __strides5 = 1
    # __shape_strides5 = [1, __strides5, __strides5, 1]
    # # 第三层卷积层padding方式
    # __padding5 = 'SAME'
    # __activation_func5 = tf.nn.relu
    # # scope名词
    # __scope_conv5 = 'concat_conv5'
    #
    # __kernel_number6 = 2048
    # # 第三层卷积核大小是7 x 7，对生成的2048个特征进行卷积，提取2048个特征
    # __shape_kernel_conv6 = [7, 7, __kernel_number4, __kernel_number5]
    # # 第三层卷积层步长
    # __strides6 = 1
    # __shape_strides6 = [1, __strides6, __strides6, 1]
    # # 第三层卷积层padding方式
    # __padding6 = 'SAME'
    # __activation_func6 = tf.nn.relu
    # # scope名词
    # __scope_conv6 = 'concat_conv6'

    def __init__(self):
        with tf.variable_scope('encoder_decoder'):
            self.conv_layer1 = ConvLayer(self.__shape_kernel_conv1, self.__kernel_number1, self.__shape_strides1,
                                         self.__padding1, self.__activation_func1, self.__scope_conv1, [], False)
            self.conv_layer2 = ConvLayer(self.__shape_kernel_conv2, self.__kernel_number2, self.__shape_strides2,
                                         self.__padding2, self.__activation_func2, self.__scope_conv2, [], False)
            # self.conv_layer3 = ConvLayer(self.__shape_kernel_conv3, self.__kernel_number3, self.__shape_strides3,
            #                              self.__padding3, self.__activation_func3, self.__scope_conv3, [], False)
            # self.conv_layer4 = ConvLayer(self.__shape_kernel_conv4, self.__kernel_number4, self.__shape_strides4,
            #                              self.__padding4, self.__activation_func4, self.__scope_conv4, [], False)
            # self.conv_layer5 = ConvLayer(self.__shape_kernel_conv5, self.__kernel_number5, self.__shape_strides5,
            #                              self.__padding5, self.__activation_func5, self.__scope_conv5, [], False)
            # self.conv_layer6 = ConvLayer(self.__shape_kernel_conv6, self.__kernel_number6, self.__shape_strides6,
            #                              self.__padding6, self.__activation_func6, self.__scope_conv6, [], False)
    def getConcat(self, data):
        conv1 = self.conv_layer1.conv(data)
        conv2 = self.conv_layer2.conv(conv1)
        #conv3 = self.conv_layer3.conv(conv2)
        #pool = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #conv4 = self.conv_layer4.conv(pool)
        #conv5 = self.conv_layer5.conv(conv4)
        #conv6 = self.conv_layer5.conv(conv5)
        return conv2
