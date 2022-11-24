
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import openslide

import cv2

parser = argparse.ArgumentParser(description='Get tissue mask of WSI and save'
                                             ' it in npy format')
parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the WSI file')
parser.add_argument('npy_path', default=None, metavar='NPY_PATH', type=str,
                    help='Path to the output npy mask file')
parser.add_argument('--level', default=6, type=int, help='at which WSI level'
                                                         ' to obtain the mask, default 6')
parser.add_argument('--RGB_min', default=50, type=int, help='min value for RGB'
                                                            ' channel, default 50')
wsi_name='430252'
def find_max_region(a):

    mask_sel = a.copy()
    contours, hierarchy = cv2.findContours(mask_sel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
    max_idx = np.argmax(area)
    for k in range(len(contours)):
        if k != max_idx:
            cv2.fillPoly(mask_sel, [contours[k]], 0)
    return mask_sel

def run():
    logging.basicConfig(level=logging.INFO)
    slide = openslide.open_slide('E:\\'+wsi_name+'.svs')
    thumb = slide.get_thumbnail(slide.level_dimensions[2])
    thumb = np.array(thumb)[:, :, :3]
    thumb = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
    thumb = cv2.threshold(thumb, 215, 255, cv2.THRESH_BINARY_INV)[1]
    thumb = cv2.GaussianBlur(thumb, (5, 5), 0)
    nd_two_img = np.array(thumb).astype(np.uint8) * 255
    mask_img = find_max_region(nd_two_img)
    contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for k in range(len(contours)):
        cv2.fillPoly(mask_img, [contours[k]], 1)
    np.save('D:/'+wsi_name+'Tumor_001_tissue.npy', mask_img)
    plt.imsave("D:/"+wsi_name+"tissue_segment.png", mask_img)
    plt.imshow(mask_img)

if __name__ == '__main__':
    run()
