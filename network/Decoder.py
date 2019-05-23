import tensorflow as tf
from network.base.ConvolutionLayer import ConvLayer

class Decoder:

    # 第一层卷积核个数
    __kernel_number1 = 512
    # 卷积核大小是3 x 3，对Y单通道进行卷积，提取64个特征
    __shape_kernel_conv1 = [5, 5, __kernel_number1, 2 * __kernel_number1]
    # 第一层卷积层步长
    __strides1 = 2
    __shape_strides1 = [1, __strides1, __strides1, 1]
    # 第一层卷积层padding方式
    __padding1 = 'SAME'
    __activation_func1 = tf.nn.relu
    # scope名称
    __scope_conv1 = 'deconv1'

    # 第二层卷积核个数
    __kernel_number2 = 256
    # 卷积核大小是5 x 5，对生成的16个特征进行卷积，提取32个特征
    __shape_kernel_conv2 = [5, 5, __kernel_number2, __kernel_number1]
    # 第二层卷积层步长
    __strides2 = 2
    __shape_strides2 = [1, __strides2, __strides2, 1]
    # 第二层卷积层padding方式
    __padding2 = 'SAME'
    __activation_func2 = tf.nn.relu
    # scope名词
    __scope_conv2 = 'deconv2'

    # 第三层卷积核个数
    __kernel_number3 = 128
    # 卷积核大小是5 x 5，对生成的16个特征进行卷积，提取33个特征
    __shape_kernel_conv3 = [4, 4, __kernel_number3, __kernel_number2]
    # 第三层卷积层步长
    __strides3 = 2
    __shape_strides3 = [1, __strides3, __strides3, 1]
    # 第三层卷积层padding方式
    __padding3 = 'SAME'
    __activation_func3 = tf.nn.relu
    # scope名词
    __scope_conv3 = 'deconv3'

    # 第二层卷积核个数
    __kernel_number4 = 32
    # 卷积核大小是5 x 5，对生成的16个特征进行卷积，提取44个特征
    __shape_kernel_conv4 = [3, 3, __kernel_number4, __kernel_number3]
    # 第二层卷积层步长
    __strides4 = 2
    __shape_strides4 = [1, __strides4, __strides4, 1]
    # 第二层卷积层padding方式
    __padding4 = 'SAME'
    __activation_func4 = tf.nn.relu
    # scope名词
    __scope_conv4 = 'deconv4'

    # 第二层卷积核个数
    __kernel_number5 = 1
    # 卷积核大小是5 x 5，对生成的16个特征进行卷积，提取1个特征
    __shape_kernel_conv5 = [3, 3, __kernel_number5, __kernel_number4]
    # 第二层卷积层步长
    __strides5 = 2
    __shape_strides5 = [1, __strides5, __strides5, 1]
    # 第二层卷积层padding方式
    __padding5 = 'SAME'
    __activation_func5 = tf.nn.relu
    # scope名词
    __scope_conv5 = 'deconv5'
    
    def __init__(self, batch_size, image_height, image_width):
        with tf.variable_scope('decoder'):
            output_shape1 = [batch_size, 48, 72, self.__kernel_number1]
            output_shape2 = [batch_size, 96, 144, self.__kernel_number2]
            output_shape3 = [batch_size, 192, 288, self.__kernel_number3]
            output_shape4 = [batch_size, 384, 576, self.__kernel_number4]
            output_shape5 = [batch_size, 768, 1152, self.__kernel_number5]

            self.deconv_layer1 = ConvLayer(self.__shape_kernel_conv1, self.__kernel_number1, self.__shape_strides1,
                                    self.__padding1, self.__activation_func1, self.__scope_conv1, output_shape1, True)
            self.deconv_layer2 = ConvLayer(self.__shape_kernel_conv2, self.__kernel_number2, self.__shape_strides2,
                                    self.__padding2, self.__activation_func2, self.__scope_conv2, output_shape2, True)
            self.deconv_layer3 = ConvLayer(self.__shape_kernel_conv3, self.__kernel_number3, self.__shape_strides3,
                                    self.__padding3, self.__activation_func3, self.__scope_conv3, output_shape3, True)
            self.deconv_layer4 = ConvLayer(self.__shape_kernel_conv4, self.__kernel_number4, self.__shape_strides4,
                                    self.__padding4, self.__activation_func4, self.__scope_conv4, output_shape4, True)
            self.deconv_layer5 = ConvLayer(self.__shape_kernel_conv5, self.__kernel_number5, self.__shape_strides5,
                                           self.__padding5, self.__activation_func5, self.__scope_conv5, output_shape5,
                                           True)

    def decode(self, images):
        out = images
        out = self.deconv_layer1.conv(out)
        out = self.deconv_layer2.conv(out)
        out = self.deconv_layer3.conv(out)
        out = self.deconv_layer4.conv(out)
        out = self.deconv_layer5.conv(out)
        return out
