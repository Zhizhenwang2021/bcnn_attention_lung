import numpy as np
from torch.utils.data import Dataset
import openslide
import PIL
from torch import nn
from torchvision import models
import glob
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image
import cv2
import tqdm

level = 0

def get_bbox_from_mask_image(tissue_mask):
    grayscale_image = cv2.cvtColor(tissue_mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(grayscale_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    return bounding_boxes

def get_samples_of_patch_starting_points(tissue_mask ):
    img = tissue_mask.copy()
    bounding_boxes = get_bbox_from_mask_image(tissue_mask)
    list_starting_points = []
    max = 0
    X = []
    Y = []
    for x, y, w, h in bounding_boxes:
        if (w >= max):
            max = w
    for x, y, w, h in bounding_boxes:
        if (w == max):
            print("w",w)
            print("h", h)
            k = 8
            X = range(x, x + w, k)
            Y = range(y, y + h, k)
    for row_starting_point in X:
        for col_starting_point in Y:
            list_starting_points.append((row_starting_point, col_starting_point))
    return list_starting_points

class WSIPatchDataset(Dataset):

    def __init__(self,wsi_path, mask_path,mask_png_path, transforms, image_size,crop_size, normalize=True, flip='NONE', rotate='NONE'):
        self._transforms = transforms
        self._mask_path = mask_path
        self._mask_png_path=mask_png_path
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self._wsi_path = wsi_path
        self._mask_path = mask_path
        self._image_size = image_size
        self._crop_size = crop_size
        self._pre_process()

    def _pre_process(self):
        self._slide = openslide.OpenSlide(self._wsi_path)
        X_slide, Y_slide = self._slide.level_dimensions[0]
        height1, width1 = self._slide.level_dimensions[2]
        self._downsamples =round(X_slide / height1)
        self._cords=[]
        self._tissue_mask = cv2.imread(self._mask_png_path, 1)
        self._mask_cords =get_samples_of_patch_starting_points(self._tissue_mask)
        for i in range(len(self._mask_cords)):
            x , y=self._mask_cords[i]
            m = int(self._image_size / self._downsamples)
            s = int(0.70 * m * m * 255)
            if np.sum(self._tissue_mask[y:y + m, x:x + m]) > s:
                self._cords.append((x, y))
        self._idcs_num = len(self._cords)

    def __len__(self):
        return self._idcs_num

    def __getitem__(self, idx):
        x, y = self._cords[idx]
        # print(idx)
        img = self._slide.read_region(
            (int(x * self._downsamples), int(y * self._downsamples)), level, (self._image_size, self._image_size)).convert(
            'RGB')
        img = img.resize((self._crop_size, self._crop_size), Image.BILINEAR)
        img = self._transforms(img)
        return (img, y, x)

import sys
import os
import argparse
import logging
import json
import time
import options
from ResBcnn import BCNN as resbcnn
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import models
from torch import nn
import matplotlib.pyplot as plt

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser(description='Get the probability map of tumor'
                                             ' patch predictions given a WSI')
parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('ckpt_path', default=None, metavar='CKPT_PATH', type=str,
                    help='Path to the saved ckpt file of a pytorch model')
parser.add_argument('cnn_path', default=None, metavar='CNN_PATH', type=str,
                    help='Path to the config file in json format related to'
                         ' the ckpt file')
parser.add_argument('mask_path', default=None, metavar='MASK_PATH', type=str,
                    help='Path to the tissue mask of the input WSI file')
parser.add_argument('probs_map_path', default=None, metavar='PROBS_MAP_PATH',
                    type=str, help='Path to the output probs_map numpy file')
parser.add_argument('--GPU', default='0', type=str, help='which GPU to use'
                                                         ', default 0')
parser.add_argument('--num_workers', default=5, type=int, help='number of '
                                                               'workers to use to make batch, default 5')
parser.add_argument('--eight_avg', default=0, type=int, help='if using average'
                                                             ' of the 8 direction predictions for each patch,'
                                                             ' default 0, which means disabled')
wsi_name='396635'
wsi_path="E:\\"+wsi_name+".svs"
npy_path = "D:\\"+wsi_name+"Tumor_001_tissue.npy"
mask_path= 'D:\\'+wsi_name+'tissue_segment.png'
flip = 'NONE'
rotate = 'NONE'
image_size=512
crop_size=224

train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

dataset = WSIPatchDataset(wsi_path, npy_path,mask_path, train_transforms, image_size,crop_size, normalize=True, flip=flip, rotate=rotate)

def get_probs_map(model1, dataloader):
    probs_map = np.zeros(dataloader.dataset._tissue_mask.shape)
    probs_map[:][:] = np.array([255, 255, 255])
    num_batch = len(dataloader)
    count = 0
    time_now = time.time()
    color_last =np.array([0,0, 0])
    cords_last = (0,0)
    probability_last=0.0
    for (data, x_mask, y_mask) in dataloader:
        data1 = Variable(data.cuda(async=True), volatile=True)
        output = model1(data1)
        for i in range(output.shape[0]):
            flag = 0
            if max(output[i]) < probability_last:
                flag = 1
            out = output[i].tolist().index(max(output[i]))
            x = int(x_mask[i])
            y = int(y_mask[i])
            if out == 1:
                probs_map[x:x + 14, y:y + 14] = np.array([28, 145, 193])
            if out == 2:
                probs_map[x:x + 14, y:y + 14] = np.array([0, 255, 255])
            if out == 3:
                probs_map[x:x + 14, y:y + 14] = np.array([255, 128, 0])
            if out == 4:
                probs_map[x:x + 14, y:y + 14] = np.array([0, 0, 0])
            if out == 5:
                probs_map[x:x + 14, y:y + 14] = np.array([64, 255, 0])
            if out == 6:
                probs_map[x:x + 14, y:y + 14] = np.array([255, 255, 0])
            if out == 7:
                probs_map[x:x + 14, y:y + 14] = np.array([0, 128, 255])
            if out == 8:
                probs_map[x:x + 14, y:y + 14] = np.array([0, 0, 255])
            if out == 9:
                probs_map[x:x + 14, y:y + 14] = np.array([255, 0, 128])
            if flag == 1:
                probs_map[cords_last[0]:cords_last[0] + 14, cords_last[1]:cords_last[1] + 14] = color_last
            color_last = probs_map[x:x + 14, y:y + 14]
            probability_last = max(output[i])
            cords_last = (x, y)
        count += 1
        time_spent = time.time() - time_now
        time_now = time.time()
        logging.info(
            '{}, flip : {}, rotate : {}, batch : {}/{}, Run Time : {:.2f}'
                .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), dataloader.dataset._flip,
                dataloader.dataset._rotate, count, num_batch, time_spent))
    return probs_map

def make_dataloader(opt, flip='NONE', rotate='NONE'):
    batch_size =  8
    num_workers = 0
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    return dataloader

def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in opt.train['gpus'])
    logging.basicConfig(level=logging.INFO)
    model1_path = opt.test['fine_tune_model1']
    model1 = resbcnn()
    checkpoint1 = torch.load(model1_path)
    model1.load_state_dict(checkpoint1['state_dict'])
    model1 = model1.cuda().eval()
    dataloader = make_dataloader(
        opt, flip='NONE', rotate='NONE')
    probs_map = get_probs_map(model1, dataloader)
    probs_map = probs_map[:, :, ::-1]
    plt.imshow(probs_map.astype(np.uint8))
    plt.imsave("D:\\ResNetBcnn_"+wsi_name+".png", probs_map.astype(np.uint8))
    plt.show()

if __name__ == '__main__':
    from options import Options
    opt = Options(isTrain=True)
    run(opt)



