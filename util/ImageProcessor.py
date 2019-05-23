import numpy as np
import cv2
import glob
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
import cv2
import imageio



class ImageProcessor:

    def __convertRGBtoYCbCr(self, im):
        xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
        ycbcr = im.dot(xform.T)
        ycbcr[:, :, [1, 2]] += 128
        return ycbcr

    def convertYCbcrToRGB(self, im):
        xform = np.array([[1, 0, 1.402], [1, -.34414, -.71414], [1, 1.772, 0]])
        rgb = im.astype(np.float)
        rgb[:, :, [1, 2]] -= 128
        rgb = rgb.dot(xform.T)
        np.putmask(rgb, rgb > 255, 255)
        np.putmask(rgb, rgb < 0, 0)
        return rgb

    def getYCrbrImage(self, imagePath):
        im = cv2.imread(imagePath)
        im = im[:, :, (2, 1, 0)]
        results = self.__convertRGBtoYCbCr(im)
        return results

    def __getYChannel(self, imagePath):
        ycbcr_image = self.getYCrbrImage(imagePath)
        y_channel, cb_channel, cr_channel = self.splitImage(ycbcr_image)
        result = y_channel.reshape([len(y_channel), len(y_channel[0]), 1])
        return result.astype(np.float)

    def splitImage(self, image):
        return cv2.split(image)


    def getYChannelOfScenes(self, imageDirPathList, suffixList):
        image_list_list = []
        for suffix in suffixList:
            image_list_list.append([])

        for scene_path in imageDirPathList:
            # img_path_list = glob.glob(scene_path + '/*.png')
            # img_path_list.sort()
            for i in range(len(suffixList)):
                temp_img = cv2.imread(scene_path + suffixList[i], flags=cv2.IMREAD_ANYDEPTH)
                # print("temp_img:", temp_img.shape)
                y_channel = self.splitImage(temp_img)[0]
                y_channel = y_channel.reshape([len(y_channel), len(y_channel[0]), 1])
                image_list_list[i].append(y_channel)

        result = []
        for image_list in image_list_list:
            result.append(np.stack(image_list))
        return result

    def save_image(self, path, data):
        cv2.imwrite(path, data)

    def merge(self, images):
        shape = images[0].shape
        height = shape[0]
        width = shape[1]
        images[0] = images[0].reshape([height, width, 1])
        images[1] = images[1].reshape([height, width, 1])
        images[2] = images[2].reshape([height, width, 1])
        return cv2.merge(images)

    def generate_training_data(self, imageDirPathList):
        merge_mertens = cv2.createMergeMertens()
        for scene_path in imageDirPathList:
            img_path_list = glob.glob(scene_path + '/input*.ppm')
            img_path_list.sort()
            cnt = 0
            temp_image_list = []
            for image_path in img_path_list:
                im = cv2.imread(image_path, flags=cv2.IMREAD_ANYDEPTH)
                y_channel = self.__getYChannel(image_path)
                image_str = scene_path + '/exposure' + str(cnt) + '.png'
                # y_channel = cv2.resize(y_channel, (512, 512), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(image_str, y_channel)
                print(image_str + ' has been generated!')
                temp_image_list.append(im)
                cnt+=1
            # rgb_gt = merge_mertens.process(temp_image_list)
            # rgb_gt*=255
            # rgb_gt_path = scene_path + '/rgb_gt.png'
            # cv2.imwrite(rgb_gt_path, rgb_gt)
            gt = cv2.imread(scene_path + '/GT(clamp).hdr', flags=cv2.IMREAD_ANYDEPTH)
            tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
            ldrDrago = tonemapDrago.process(gt)
            ldrDrago = 3 * ldrDrago
            cv2.imwrite(scene_path + "/rgb_gt.png", ldrDrago * 255)
            y_channel_of_gt = self.__getYChannel(scene_path+'/rgb_gt.png')
            gt_path = scene_path + '/gt.png'
            # y_channel_of_gt = cv2.resize(y_channel_of_gt, (512, 512), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(gt_path, y_channel_of_gt)
            print(gt_path + ' has been generated!')
