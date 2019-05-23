import tensorflow as tf


class ConvLayer:

    def __init__(self, shape_kernel, kernel_number, shape_strides, padding, activation_func, scope, output_shape, deconv):
        self.__shape_kernel = shape_kernel
        self.__kernel_number = kernel_number
        self.__shape_strides = shape_strides
        self.__padding = padding
        self.__activation_func = activation_func
        self.__scope = scope
        self.__output_shape = output_shape
        self.__deconv = deconv

    def __weight_variable(self, shape):
        with tf.variable_scope(self.__scope):
            initial = tf.truncated_normal(shape, stddev=0.1, name= self.__scope + 'kernel')
        return tf.Variable(initial)

    def __bias_variable(self, shape):
        with tf.variable_scope(self.__scope):
            initial = tf.constant(0.1, shape=shape, name=self.__scope + 'bias')
        return tf.Variable(initial)

    def __conv2d(self, images, weights, shape_strides, padding):
        return tf.nn.conv2d(images, weights, strides=shape_strides, padding=padding)

    def __deconv2d(self, images, weights, output_shape, shape_strides, padding):
        return tf.nn.conv2d_transpose(images, weights, output_shape, shape_strides, padding)

    def conv(self, images):
        weights = self.__weight_variable(self.__shape_kernel)
        bias = self.__bias_variable([self.__kernel_number])
        if self.__deconv:
            output_of_kernel = self.__deconv2d(images, weights, self.__output_shape, self.__shape_strides,
                                               self.__padding)
        else:
            output_of_kernel = self.__conv2d(images, weights, self.__shape_strides, self.__padding) + bias
        output_of_activation = tf.nn.relu(output_of_kernel)
        return output_of_activation
