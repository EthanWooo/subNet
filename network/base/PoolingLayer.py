import tensorflow as tf


class PoolingLayer:

    def __init__(self, output_of_conv, kernel_shape, strides, padding, pool_func):
        self.output_of_conv = output_of_conv
        self.kernel_shape = kernel_shape
        self.strides = strides
        self.padding = padding
        self.pool_func = pool_func

    def pooling(self):
        return self.pool_func(self.output_of_conv, ksize=self.kernel_shape, strides=self.strides, padding=self.padding)
