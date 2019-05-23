import sys
import os
import random
import numpy as np
import scipy.io as sio

import tensorflow as tf
from network.Encoder import Encoder
from network.Decoder import Decoder
from network.Strategy import Strategy
from util.ImageProcessor import ImageProcessor
from network.WeightedFusion import WeightedFusion
from datetime import datetime


class DeepFuseNetwork:
    __batch_size = 1
    __image_height = 768
    __image_width = 1152
    __image_channel = 1

    def __init__(self):
        self.__encoder = Encoder()
        self.__decoder = Decoder(self.__batch_size, self.__image_height, self.__image_width)

    # 输入的是图像的tensor
    def __fuse(self, exposure1, exposure2, exposure3, flow1, flow2):
        enc_1 = self.__encoder.encode(exposure1)
        enc_2 = self.__encoder.encode(exposure2)
        enc_3 = self.__encoder.encode(exposure3)
        target_features = Strategy(enc_1, enc_2, enc_3, flow1, flow2)
        generated_img = self.__decoder.decode(target_features)
        return generated_img

    # 输入的是图像numpy的list，每个元素是height * width
    def __generate_fused_Y_channel(self, image1, image2, image3, flow_array1, flow_array2, model_path):
        image1 = image1.reshape([1, image1.shape[0], image1.shape[1], 1])
        image2 = image2.reshape([1, image2.shape[0], image2.shape[1], 1])
        image3 = image3.reshape([1, image3.shape[0], image3.shape[1], 1])
        with tf.Graph().as_default(), tf.Session() as sess:
            exposure_h = tf.placeholder(tf.float32, shape=image1.shape, name='exposure_h')
            exposure_r = tf.placeholder(tf.float32, shape=image2.shape, name='exposure_r')
            exposure_l = tf.placeholder(tf.float32, shape=image3.shape, name='exposure_l')
            flow_h2r = tf.placeholder(tf.float32, shape=(1, 96, 144, 256), name='flow_h2r')
            flow_l2r = tf.placeholder(tf.float32, shape=(1, 96, 144, 256), name='flow_l2r')
            output = self.__fuse(exposure_h, exposure_r, exposure_l, flow_h2r, flow_l2r)
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            output = sess.run(output, feed_dict={exposure_h: image1,
                                                 exposure_r: image2,
                                                 exposure_l: image3,
                                                 flow_h2r: flow_array1,
                                                 flow_l2r: flow_array2})
            return output

    def generate_fused_image(self, image_paths, model_path, save_path):
        image_processor = ImageProcessor()
        weighted_fusion = WeightedFusion()
        image1 = image_processor.getYCrbrImage(image_paths[0])
        image2 = image_processor.getYCrbrImage(image_paths[1])
        image3 = image_processor.getYCrbrImage(image_paths[2])
        flow1 = sio.loadmat(image_paths[3])
        flow1 = flow1['future']
        flow1 = flow1[np.newaxis, :, :, :]
        flow2 = sio.loadmat(image_paths[4])
        flow2 = flow2['future']
        flow2 = flow2[np.newaxis, :, :, :]

        fused_y_channel = self.__generate_fused_Y_channel(image1[:, :, 0], image2[:, :, 0], image3[:, :, 0],
                                                          flow1, flow2, model_path)
        fused_y_channel = fused_y_channel.reshape([image2.shape[0], image2.shape[1]])

        fused_cb_channel = weighted_fusion.fuse(image1[:, :, 1], image2[:, :, 1], image3[:, :, 1], 128)
        fused_cr_channel = weighted_fusion.fuse(image1[:, :, 2], image2[:, :, 2], image3[:, :, 1], 128)

        fused_image = np.stack([fused_y_channel, fused_cb_channel, fused_cr_channel], axis=2)
        result_image = image_processor.convertYCbcrToRGB(fused_image)
        result_image = result_image[:, :, (2, 1, 0)]
        image_processor.save_image(save_path, result_image)

    def generate_fused_Y_channel_image(self, image_paths, model_path, save_path):
        image_processor = ImageProcessor()
        image1 = image_processor.getYCrbrImage(image_paths[0])
        image2 = image_processor.getYCrbrImage(image_paths[1])
        image3 = image_processor.getYCrbrImage(image_paths[2])
        flow1 = sio.loadmat(image_paths[3])
        flow1 = flow1['future']
        flow1 = flow1[np.newaxis, :, :, :]
        flow2 = sio.loadmat(image_paths[4])
        flow2 = flow2['future']
        flow2 = flow2[np.newaxis, :, :, :]

        fused_y_channel = self.__generate_fused_Y_channel(image1[:, :, 0], image2[:, :, 0], image3[:, :, 0],
                                                          flow1, flow2, model_path)
        fused_y_channel = fused_y_channel.reshape([image2.shape[0], image2.shape[1]])
        image_processor.save_image(save_path, fused_y_channel)

    def train(self, image_paths, save_path, epoches, batch_size, learning_rate):
        start_time = datetime.now()
        print("EPOCHES   : ", epoches)
        print("BATCH_SIZE: ", batch_size)
        num_scenes = len(image_paths)
        # num_imgs = 100
        image_paths = image_paths[:num_scenes]
        mod = num_scenes % batch_size

        print('Train images number %d.' % num_scenes)
        print('Train images samples %s.' % str(num_scenes / batch_size))

        if mod > 0:
            print('Train set has been trimmed %d samples...\n' % mod)
            image_paths = image_paths[:-mod]

        # get the traing image shape
        image_shape = (batch_size, self.__image_height, self.__image_width, self.__image_channel)
        flow_shape = (batch_size, 96, 144, 256)

        # create the graph
        image_procsssor = ImageProcessor()
        weighted_fusion = WeightedFusion()
        with tf.Graph().as_default(), tf.Session() as sess:
            original = tf.placeholder(tf.float32, shape=image_shape, name='original')
            exposure_h = tf.placeholder(tf.float32, shape=image_shape, name='exposure_h')
            exposure_r = tf.placeholder(tf.float32, shape=image_shape, name='exposure_r')
            exposure_l = tf.placeholder(tf.float32, shape=image_shape, name='exposure_l')
            flow_h2r = tf.placeholder(tf.float32, shape=flow_shape, name='flow_h2r')
            flow_l2r = tf.placeholder(tf.float32, shape=flow_shape, name='flow_l2r')
            print('exposure_h  :', exposure_h.shape)
            print('exposure_r  :', exposure_r.shape)
            print('exposure_l  :', exposure_l.shape)
            print('original:', original.shape)

            generated_img = self.__fuse(exposure_h, exposure_r, exposure_l, flow_h2r, flow_l2r)
            tf.print(generated_img, [generated_img.shape])

            print('generate:', generated_img.shape)

            ssim_value = tf.image.ssim(original, generated_img, max_val=255)
            loss1 = 1 - ssim_value
            loss2 = tf.nn.l2_loss(generated_img-original)/(self.__image_height*self.__image_width)
            loss = 1*loss1+0*loss2
            #loss = 0.8*loss1*loss2+0.2*loss2
            #generated_tm = tf.log1p(generated_img)
            #original_tm = tf.log1p(original)
            #loss = tf.losses.mean_squared_error(original, generated_img)
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

            saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
            if os.path.exists(save_path + '/test_model.ckpt.meta'):
                print('恢复模型:', save_path + '/test_model.ckpt')
                saver.restore(sess, save_path + '/test_model.ckpt')
            else:
                print('不恢复模型:', save_path + '/test_model.ckpt')
                sess.run(tf.global_variables_initializer())

            # ** Start Training **

            min_loss = sys.maxsize
            max_ssim = 0
            n_batches = int(len(image_paths) // batch_size)

            elapsed_time = datetime.now() - start_time
            print('Elapsed time for preprocessing before actually train the model: %s' % elapsed_time)
            print('\nNow begin to train the model...\n')
            start_time = datetime.now()

            for epoch in range(epoches):
                random.shuffle(image_paths)
                loss_list = []
                ssim_list = []
                for batch in range(n_batches):
                    # retrive a batch of content and style images

                    now_batch_paths = image_paths[batch * batch_size:(batch * batch_size + batch_size)]
                    image_result = image_procsssor.getYChannelOfScenes(now_batch_paths,
                                                                       ['/exposure0.png', '/exposure1.png', '/exposure3.png',
                                                                        '/gt.png'])
                    exposureh_batch = image_result[0]
                    exposurer_batch = image_result[1]
                    exposurel_batch = image_result[2]
                    original_batch = image_result[3]
                    # print('original_batch shape final:', original_batch.shape)
                    flow_h2r_batch = sio.loadmat(now_batch_paths[0] + '/h2r256channal.mat')
                    flow_h2r_batch = flow_h2r_batch['future']
                    flow_h2r_batch = flow_h2r_batch[np.newaxis, :, :, :]

                    flow_l2r_batch = sio.loadmat(now_batch_paths[0] + '/l2r256channal.mat')
                    flow_l2r_batch = flow_l2r_batch['future']
                    flow_l2r_batch = flow_l2r_batch[np.newaxis, :, :, :]
                    # run the training step
                    _, _ssim_loss, _loss, _generated_img = sess.run([train_op, ssim_value, loss, generated_img],
                                                                    feed_dict={original: original_batch,
                                                                               exposure_h: exposureh_batch,
                                                                               exposure_r: exposurer_batch,
                                                                               exposure_l: exposurel_batch,
                                                                               flow_h2r: flow_h2r_batch,
                                                                               flow_l2r: flow_l2r_batch})
                    loss_list.append(_loss[0])
                    ssim_list.append(_ssim_loss[0])
                    if batch == n_batches - 1:
                        elapsed_time = datetime.now() - start_time
                        avg_loss = np.mean(loss_list)
                        avg_ssim_loss = np.mean(ssim_list)
                        print('Deep fuse==>>epoch: %d,  loss: %s,  ssim: %s,  max_loss: %s,  min_ssim: %s,  elapsed time: %s'
                              % (epoch, avg_loss, avg_ssim_loss, max(loss_list), min(ssim_list), elapsed_time))
                        if avg_loss < min_loss:
                            print('Update==>>best loss Model saved')
                            saver.save(sess, save_path + '/best_loss/test_model.ckpt')
                            min_loss = avg_loss
                        if avg_ssim_loss > max_ssim:
                            print('Update==>>best ssim Model saved')
                            saver.save(sess, save_path + '/best_ssim/test_model.ckpt')
                            max_ssim = avg_ssim_loss
                        saver.save(sess, save_path + '/test_model.ckpt')
                          
                        image1 = image_procsssor.getYCrbrImage(now_batch_paths[0] + '/inputh.ppm')
                        image2 = image_procsssor.getYCrbrImage(now_batch_paths[0] + '/inputr.ppm')
                        image3 = image_procsssor.getYCrbrImage(now_batch_paths[0] + '/inputl.ppm')
                        generated_y_channel = _generated_img.reshape([image2.shape[0], image2.shape[1]])
                        generated_cb_channel = weighted_fusion.fuse(image1[:, :, 1], image2[:, :, 1], image3[:, :, 1], 128)
                        generated_cr_channel = weighted_fusion.fuse(image1[:, :, 2], image2[:, :, 2], image3[:, :, 1], 128)
                        fused_image = np.stack([generated_y_channel, generated_cb_channel, generated_cr_channel], axis=2)
                        result_image = image_procsssor.convertYCbcrToRGB(fused_image)
                        result_image = result_image[:, :, (2, 1, 0)]
                        image_procsssor.save_image('./result.png', result_image)
                        image_procsssor.save_image('./result_y.png', generated_y_channel)
            elapsed_time = datetime.now() - start_time
            print('Done training! Elapsed time: %s' % elapsed_time)
