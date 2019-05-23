import sys
import glob
from network.DeepFuseNetwork import DeepFuseNetwork

MODEL_SAVE_PATH = './models/best_ssim/test_model.ckpt'
IMG_PATH = './images/MyTest'
RES_SAVE_PATH = './res/test'
dfn = DeepFuseNetwork()
img_path_list = glob.glob(IMG_PATH + '/*')
print(img_path_list)
for prefix in img_path_list:
    image_paths = [prefix + '/inputh.ppm',
                   prefix + '/inputr.ppm',
                   prefix + '/inputl.ppm',
                   prefix + '/h2r256channal.mat',
                   prefix + '/l2r256channal.mat']
    index = prefix.split('/')[-1]
    dfn.generate_fused_image(image_paths, MODEL_SAVE_PATH, RES_SAVE_PATH+'/result'+index+'.png')
    dfn.generate_fused_Y_channel_image(image_paths, MODEL_SAVE_PATH, RES_SAVE_PATH+'/result'+index+'_y.png')
