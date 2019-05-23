import tensorflow as tf
from network.base.ConvolutionLayer import ConvLayer

class Encoder:

    # 第一层卷积核个数
    __kernel_number1 = 16
    # 卷积核大小是3 x 3，对Y单通道进行卷积，提取8个特征[384,576,8]
    __shape_kernel_conv1 = [3, 3, 1, __kernel_number1]
    # 第一层卷积层步长
    __strides1 = 2
    __shape_strides1 = [1, __strides1, __strides1, 1]
    # 第一层卷积层padding方式
    __padding1 = 'SAME'
    __activation_func1 = tf.nn.relu
    # scope名称
    __scope_conv1 = 'conv1'
    __output_shape1 = []

    # 第二层卷积核个数
    __kernel_number2 = 32
    # 卷积核大小是5 x 5，对生成的32个特征进行卷积，提取32个特征[192,288,32]
    __shape_kernel_conv2 = [5, 5, __kernel_number1, __kernel_number2]
    # 第二层卷积层步长
    __strides2 = 2
    __shape_strides2 = [1, __strides2, __strides2, 1]
    # 第二层卷积层padding方式
    __padding2 = 'SAME'
    __activation_func2 = tf.nn.relu
    # scope名词
    __scope_conv2 = 'conv2'
    
    # 第三层卷积核个数
    __kernel_number3 = 64
    # 卷积核大小是3 x 3，对生成的33个特征进行卷积，提取64个特征[192,288,64]
    __shape_kernel_conv3 = [3, 3, __kernel_number2, __kernel_number3]
    # 第二层卷积层步长
    __strides3 = 1
    __shape_strides3 = [1, __strides3, __strides3, 1]
    # 第二层卷积层padding方式
    __padding3 = 'SAME'
    __activation_func3 = tf.nn.relu
    # scope名词
    __scope_conv3 = 'conv3'
    
    # 第四层卷积核个数
    __kernel_number4 = 128
    # 卷积核大小是2 x 2，对生成的44个特征进行卷积，提取128个特征[96,144,128]
    __shape_kernel_conv4 = [2, 2, __kernel_number3, __kernel_number4]
    # 第二层卷积层步长
    __strides4 = 2
    __shape_strides4 = [1, __strides4, __strides4, 1]
    # 第二层卷积层padding方式
    __padding4 = 'SAME'
    __activation_func4 = tf.nn.relu
    # scope名词
    __scope_conv4 = 'conv4'

    # 第五层卷积核个数
    __kernel_number5 = 256
    # 卷积核大小是2 x 2，对生成的44个特征进行卷积，提取256个特征[96,144,256]
    __shape_kernel_conv5 = [2, 2, __kernel_number4, __kernel_number5]
    # 第二层卷积层步长
    __strides5 = 1
    __shape_strides5 = [1, __strides5, __strides5, 1]
    # 第二层卷积层padding方式
    __padding5 = 'SAME'
    __activation_func5 = tf.nn.relu
    # scope名词
    __scope_conv5 = 'conv5'
    
    def __init__(self):
        with tf.variable_scope('encoder'):
            self.conv_layer1 = ConvLayer(self.__shape_kernel_conv1, self.__kernel_number1, self.__shape_strides1,
                                    self.__padding1, self.__activation_func1, self.__scope_conv1, [], False)
            self.conv_layer2 = ConvLayer(self.__shape_kernel_conv2, self.__kernel_number2, self.__shape_strides2,
                                    self.__padding2, self.__activation_func2, self.__scope_conv2, [], False)
            self.conv_layer3 = ConvLayer(self.__shape_kernel_conv3, self.__kernel_number3, self.__shape_strides3,
                                         self.__padding3, self.__activation_func3, self.__scope_conv3, [], False)
            self.conv_layer4 = ConvLayer(self.__shape_kernel_conv4, self.__kernel_number4, self.__shape_strides4,
                                         self.__padding4, self.__activation_func4, self.__scope_conv4, [], False)
            self.conv_layer5 = ConvLayer(self.__shape_kernel_conv5, self.__kernel_number5, self.__shape_strides5,
                                         self.__padding5, self.__activation_func5, self.__scope_conv5, [], False)

    def encode(self, images):
        out = images
        out = self.conv_layer1.conv(out)
        out = self.conv_layer2.conv(out)
        out = self.conv_layer3.conv(out)
        out = self.conv_layer4.conv(out)
        out = self.conv_layer5.conv(out)
        return out
