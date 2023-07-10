import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.cvlibs import param_init

from PIL import Image
import numpy as np
import time
from os import mkdir
from os.path import join, isdir
from tqdm import tqdm
import glob

from util_paddle import PSNR, GeneratorEnqueuer, DirectoryIterator_DIV2K, _load_img_array, _rgb2ycbcr
from tensorboardX import SummaryWriter

### USER PARAMS ###
EXP_NAME = "SR-LUT"
VERSION = "S"
UPSCALE = 4     # upscaling factor
VAL_DIR = './val/'      # Validation images

### A lightweight deep network ###
class SRNet(paddle.nn.Layer):
    def __init__(self, upscale=4):
        super(SRNet, self).__init__()

        self.upscale = upscale

        self.conv1 = nn.Conv2D(1, 64, [2 ,2], stride=1, padding=0, dilation=1)
        self.conv2 = nn.Conv2D(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv3 = nn.Conv2D(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv4 = nn.Conv2D(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv5 = nn.Conv2D(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv6 = nn.Conv2D(64, 1* upscale * upscale, 1, stride=1, padding=0, dilation=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale)
    def forward(self, x_in):
        B, C, H, W = x_in.shape
        x_in = paddle.reshape(x_in,[B * C, 1, H, W])

        x = self.conv1(x_in)
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        x = self.conv4(F.relu(x))
        x = self.conv5(F.relu(x))
        x = self.conv6(F.relu(x))
        x = self.pixel_shuffle(x)
        x = paddle.reshape(x,[B, C, self.upscale * (H - 1), self.upscale * (W - 1)])
        return x

model_G = SRNet(upscale=UPSCALE)
lm = paddle.load('checkpoint/{}/model_G_i200000.pdparms'.format(str(VERSION)))
model_G.set_state_dict(lm)

if not isdir('result'):
    mkdir('result')
if not isdir('result/{}'.format(str(VERSION))):
    mkdir('result/{}'.format(str(VERSION)))

#testing
with paddle.no_grad():
    model_G.eval()

    # Test for validation images
    files_gt = glob.glob(VAL_DIR + '/HR/*.png')
    files_gt.sort()
    files_lr = glob.glob(VAL_DIR + '/LR/*.png')
    files_lr.sort()

    psnrs = []
    lpips = []

    for ti, fn in enumerate(files_gt):
        # Load HR image
        tmp = _load_img_array(files_gt[ti])
        val_H = np.asarray(tmp).astype(np.float32)  # HxWxC

        # Load LR image
        tmp = _load_img_array(files_lr[ti])
        val_L = np.asarray(tmp).astype(np.float32)  # HxWxC
        val_L = np.transpose(val_L, [2, 0, 1])  # CxHxW
        val_L = val_L[np.newaxis, ...]  # BxCxHxW

        val_L = paddle.to_tensor(val_L.copy())

        # Run model
        batch_S1 = model_G(F.pad(val_L, (0, 1, 0, 1), mode='reflect'))

        batch_S2 = model_G(F.pad(paddle.rot90(val_L, 1, [2, 3]), (0, 1, 0, 1), mode='reflect'))
        batch_S2 = paddle.rot90(batch_S2, 3, [2, 3])

        batch_S3 = model_G(F.pad(paddle.rot90(val_L, 2, [2, 3]), (0, 1, 0, 1), mode='reflect'))
        batch_S3 = paddle.rot90(batch_S3, 2, [2, 3])

        batch_S4 = model_G(F.pad(paddle.rot90(val_L, 3, [2, 3]), (0, 1, 0, 1), mode='reflect'))
        batch_S4 = paddle.rot90(batch_S4, 1, [2, 3])

        batch_S = (paddle.clip(batch_S1, -1, 1) * 127 + paddle.clip(batch_S2, -1, 1) * 127)
        batch_S += (paddle.clip(batch_S3, -1, 1) * 127 + paddle.clip(batch_S4, -1, 1) * 127)
        batch_S /= 255.0

        # Output
        image_out = batch_S.numpy()
        image_out = np.clip(image_out[0], 0., 1.)  # CxHxW
        image_out = np.transpose(image_out, [1, 2, 0])  # HxWxC

        # Save to file
        image_out = ((image_out) * 255).astype(np.uint8)
        Image.fromarray(image_out).save('result/{}/{}'.format(str(VERSION), fn.split('/')[-1]))

        # PSNR on Y channel
        img_gt = (val_H * 255).astype(np.uint8)
        CROP_S = 4
        psnrs.append(PSNR(_rgb2ycbcr(img_gt)[:, :, 0], _rgb2ycbcr(image_out)[:, :, 0], CROP_S))

print('AVG PSNR: Validation: {}'.format(np.mean(np.asarray(psnrs))))
