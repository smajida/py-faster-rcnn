#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

import caffe, os, sys, cv2
import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import os

import matplotlib.image as mpimage
from termcolor import colored
import warnings

'''
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
#from skimage.transform import rotate
from scipy.ndimage import rotate
from bs4 import BeautifulSoup
import re
from shapely.geometry import box
from shapely.affinity import rotate as r
from shapely.geometry import Polygon
from matplotlib import pyplot
from descartes import PolygonPatch
import sys
import itertools
import math
import locale
import datetime
import getpass
from PIL import Image
'''



def read_txt(labelname, width, height):

    # Get the filename where to load/save labels
    # Returns empty string if not possible
    # Set the createDirs to true, if you want to create needed directories
    file = open(labelname)
    data_label = []
    for line in file:
        if not (line[0] == 'g' or line[0] == 'i'):
            line = list(line.split(" "))
            if VERBOSE:
                print(line)
            name = line[8]
            # strange thing is dota txt, difficult parameter is together with '\n' and we have to use [9][0]
            difficult = int(line[9][0])
            annotation = line[0:8]
            name, difficult, annotation, passed = read_safe([name, difficult, annotation], labelname, line, width, height)
            if passed:
                data_label.append([name, difficult, annotation])
    dff = pd.DataFrame(data=data_label, columns=['type', 'difficult', 'annotation'])
    if VERBOSE:
        print(dff)
    if len(dff) <= 0:
        warnings.warn("no objects in the label file", UserWarning)
        return
    print("txt was read")
    return dff


def read_safe(tocheck, filename, readline, width, height):
    # tocheck[0]  is name
    # tocheck[1]  is difficulty
    # tocheck[2]  is annotation
    tocheck[2] = list(map(int, tocheck[2]))
    xs = tocheck[2][0:8:2]
    ys = tocheck[2][1:9:2]
    passed = True
    if sum(1 for number in tocheck[2] if number < 0):
        # no negative is tolerated
        print(filename)
        print(readline)
        print(tocheck)
        print(width)
        print(height)
        raise AssertionError("negative, ignored!")
        # warnings.warn("negative or violated size ", UserWarning)
    elif sum(1 for number in xs if number > width + 3) > 2: #TODO 3) > 2:
        # three pixel violation is tolerated
        print(filename)
        print(readline)
        print(tocheck)
        print(width)
        print(height)
        # TODO clamp three  pixel violation to width
        #if not 'P0173' in filename and xs[3] == 2103:  # P0173 has sample harbor with 2 pixel higher than width
        #raise AssertionError("violated width with more than two pixels")
        warnings.warn("violated size with {} pixel".format(xs), UserWarning)
        passed = False
    elif sum(1 for number in ys if number > height + 3) > 2: #TODO 3) > 2:
        # three pixel violation is tolerated
        print(filename)
        print(readline)
        print(tocheck)
        print(width)
        print(height)
        # TODO clamp three pixel violation to height
        # raise AssertionError("violated height with more than two pixels")
        warnings.warn("negative or violated size ", UserWarning)
        passed = False

        # indices = [line[i]='1' for i, line_ in enumerate(line[0:8]) if line_ in '0']
    if 0 in map(int, tocheck[2]):
        for i, line_ in enumerate(tocheck[2]):
            # if sum(1 for number in tocheck[2] if number == 0) > 1:
            if sum(1 for number in xs  if number == 0) > 2:
                raise AssertionError("WARNING: many x zeroes, ignored!")
            if sum(1 for number in xs  if number == 0) > 2:
                raise AssertionError("WARNING: many y zeroes, ignored!")
            if line_ == 0:
                tocheck[2][i] = 1
                if VERBOSE:
                    print(readline)
                    print(tocheck)
                    print(width)
                    print(height)
                    print(filename)
                # raise AssertionError("WARNING: one sample has parameter equal to zero, ignored!")
                # no warning for at most two zeros as there are quite some in dota dataset.
                # warnings.warn("One sample has parameter annotation equal to zero, ignored! ", UserWarning)
                # continue

    xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
    width_obj = xmax - xmin
    height_obj = ymax - ymin
    if not (2 < width_obj < width - 1 and 5 < height_obj < height - 1): #TODO or 10<=
        print(filename)
        print(readline)
        print(width_obj)
        print(height_obj)
        print(width)
        print(height)
        # raise AssertionError("double check width and height violated image size, ignored!")
        # TODO I noticed the minium width is 3 and also many of not passed ones were with width or height of 4 or 5.
        # maybe we could decrease the lower bound to 4.
        warnings.warn("WARNING: double check width and height violated image size, ignored!", UserWarning)
        passed = False
    return tocheck[0], tocheck[1], tocheck[2], passed




def magic_test(image, params, width, height):

    kernel_size, c_stride, r_stride = [params['kernel size'], params['c_stride'], params['r_stride']]
    if VERBOSE:
        print(kernel_size, c_stride, r_stride)

    image_dim_0, image_dim_1 = height, width   # col == width , row == height
    cropped = []

    if image_dim_1 <= kernel_size and image_dim_0 <= kernel_size:
        x, y = 0, 0
        print("smaller than kernel size in both directions")
        cropped.append((image, x, y))
        print(colored("cropped append finished it should be written once!", "green"))

    elif image_dim_0 < kernel_size:
        col = 0
        row = 0
        y = 0
        print("smaller than kernel size in height")
        print(colored("image_dim_0 < kernel_size:", 'red'))
        if VERBOSE:
            print(colored('magic: row + kernel_size...', 'red'))
        while col + kernel_size < image_dim_1:

            if VERBOSE:
                print(colored('magic: col + kernel_size...', 'blue'))
            if VERBOSE:
                print("normal patch is {}".format(np.shape(image[:, col:col + kernel_size])))
            x = col
            cropped.append((image[:, col:col + kernel_size], x, y))
            col += c_stride

        if VERBOSE:
            print("last patch in the column is {}".format(np.shape(image[:, image_dim_1 - kernel_size:])))
        x = image_dim_1 - kernel_size
        cropped.append((image[:, image_dim_1 - kernel_size:], x, y))

    elif image_dim_1 < kernel_size:
        row = 0
        col = 0
        x = 0
        print("smaller than kernel size in width")
        print(colored("image_dim_1 < kernel_size:", 'red'))
        while row + kernel_size < image_dim_0:

            if VERBOSE:
                print(colored('magic: col + kernel_size...', 'red'))
            if VERBOSE:
                print(colored('magic: col + kernel_size...', 'blue'))

            if VERBOSE:
                print("normal patch is {}".format(np.shape(image[row:row + kernel_size, :])))
            y = row
            cropped.append((image[row:row + kernel_size, :], x, y))
            row += r_stride

        if VERBOSE:
            print("last patch in the column is {}".format(np.shape(image[image_dim_0 - kernel_size:, col:image_dim_1 + 1])))
        y = image_dim_0 - kernel_size
        #TODO col:image_dim_1 + 1 was changed to : according to previous case in height, but it is not yet changes in training magic
        cropped.append((image[image_dim_0 - kernel_size:, :], x, y))

    else:
        row = 0
        print("smaller than kernel size in non of directions")
        while row + kernel_size < image_dim_0:
            col = 0
            if VERBOSE:
                print(colored('magic: row + kernel_size...', 'red'))
            while col + kernel_size < image_dim_1:
                if VERBOSE:
                    print(colored('magic: col + kernel_size...', 'blue'))

                if VERBOSE:
                    print("normal patch is {}".format(np.shape(image[row:row + kernel_size, col:col + kernel_size])))
                x= col
                y= row
                cropped.append((image[row:row + kernel_size, col:col + kernel_size], x, y))
                col += c_stride

            if VERBOSE:
                print("last patch in the column is {}".format(
                    np.shape(image[row:row + kernel_size, image_dim_1 - kernel_size:])))
            x = image_dim_1 - kernel_size
            y = row
            cropped.append((image[row:row + kernel_size, image_dim_1 - kernel_size:], x, y))
            row += r_stride
        col = 0
        while col + kernel_size < image_dim_1:
            if VERBOSE:
                print(colored('magic: col + kernel_size...', 'red'))
            if VERBOSE:
                print("last patches in the row is {}".format(
                    np.shape(image[image_dim_0 - kernel_size:, col:col + kernel_size])))
            x = col
            y = image_dim_0 - kernel_size
            cropped.append((image[image_dim_0 - kernel_size:, col:col + kernel_size], x, y))
            col += c_stride

        if VERBOSE:
            print("last patch is {}".format(np.shape(image[image_dim_0 - kernel_size:, image_dim_1 - kernel_size:])))
        x = image_dim_1 - kernel_size
        y = image_dim_0 - kernel_size
        cropped.append((image[image_dim_0 - kernel_size:, image_dim_1 - kernel_size:], x, y))
    return cropped


def demoCombined(net, im):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(args.data_dir, image_name)
    #im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    _scores, _boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, _boxes.shape[0])
    #print(scores)
    #print(boxes)
    return _scores, _boxes

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Proto directory',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--mddir', dest='model_dir', help='model directory',
                        choices=NETS.keys(), default='output/faster_rcnn_end2end')
    parser.add_argument('--dtdir', dest='data_dir', help='images directory',
                        choices=NETS.keys(), default='../preprocessing/DOTA/dota/original/test/images')

    args = parser.parse_args()

    return args




# parsign argument is needed.

#if sys.argv[0] and sys.argv[1]:
#    INT_DIR = sys.argv[0]
#    OUT_DIR = sys.argv[1]
#else:

DEBUG = False
if DEBUG:
    dataset_name = 'dota/original/check'
    INT_DIR = './'+ dataset_name
    OUT_DIR = './'+ dataset_name
else:
    dataset_name = '../preprocessing/DOTA/dota/original'
    state = '' #/separate/grayscale
    INT_DIR = './'+ dataset_name + state
    OUT_DIR = './'+ dataset_name + '/' + 'patchs_' + state

VERBOSE = False


CLASSES = ('__background__', # always index 0
              'large-vehicle', 'swimming-pool', 'helicopter', 'bridge', 'plane',
              'ship', 'soccer-ball-field', 'basketball-court','ground-track-field',
             'small-vehicle', 'harbor', 'baseball-diamond','tennis-court', 'roundabout', 'storage-tank')

CLASSES_color = {'__background__':[0, 0, 0], # always index 0
              'large-vehicle':[255, 0, 0], 'swimming-pool':[0, 255, 0], 'helicopter':[0, 0, 255],
                 'bridge':[255, 0, 255], 'plane':[255, 255, 0],
              'ship':[0, 255, 255], 'soccer-ball-field':[128, 0, 0], 'basketball-court':[0, 12, 255],
                 'ground-track-field':[255, 0, 255],
              'small-vehicle':[255, 0, 255], 'harbor':[255, 0, 255], 'baseball-diamond':[255, 0, 255],
                 'tennis-court':[255, 0, 255], 'roundabout':[255, 0, 255], 'storage-tank':[255, 0, 255]}

NETS = {'vgg16': ('VGG16',
                  'vgg16_faster_rcnn_iter_20000.caffemodel'),
        'zf': ('ZF',
                  'zf_faster_rcnn_iter_10000.caffemodel')}


# TODO: add the condition that if the size is smaller than kernel_size, keep that size as kernel
#mode = dict(trainval={'kernel size': 1000, 'c_stride': 500, 'r_stride': 500},
#            train={'kernel size': 1000, 'c_stride': 500, 'r_stride': 500},
#            val={'kernel size': 1000, 'c_stride': 700, 'r_stride': 700},
#            test={'kernel size': 1000, 'c_stride': 700, 'r_stride': 700})
# be careful val should be added at the end of train number which means first train should be read. make sure this happens!!!!
mode = dict(test={'kernel size': 1800, 'c_stride': 500, 'r_stride': 500})
#mode = dict(test={'kernel size': 1000, 'c_stride': 700, 'r_stride': 700})

data = ['images', 'labels']
aug = ['main', '90', '180', '270']
flip_aug = False
rotate_aug = False
vis = False
dpi = 80 # matplotlib.image changes the reosolution when imsave. dpi prevents this.
EPS = 0.7























def nms_vis_patch(patch_boxes, patch_scores, im, x, y, counter):
    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]
    _, ax = plt.subplots(figsize=(12, 12))
    plt.imshow(im)
    ax = plt.gca()
    for cls_ind, class_name in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background

        cls_boxes = patch_boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = patch_scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
            continue
        colorVal = scalarMap.to_rgba(values[cls_ind])
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor=colorVal, linewidth=3.5)
            )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:3f}'.format(class_name, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')

    ax.set_title(('object detections with '
                  'p(object | box) >= {:.1f}').format(CONF_THRESH), fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    #plt.show(block=False)

    if CONTAIN_OBJ:
        plt.axis('off')
        plt.tight_layout()
        directory = args.model_dir + 'prediction'
        if not os.path.exists(directory):
            os.makedirs(directory)
        # print('printing to {}'.format(directory + '/' + 'pred_' + _name))
        # plt.figure(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
        plt.savefig(directory + '/' + 'pred_' + _name + '_' + str(count) + '_' + str(x) + '_' + str(y) + '.jpg')
        plt.close()
    plt.close('all')

def nms_vis_all(all_boxes, all_scores, image):
    # Visualize detections for each class

    image = image[:, :, (2, 1, 0)]

    _, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(im, aspect='equal')

    plt.imshow(image)

    ax = plt.gca()
    # ax.set_autoscale_on(True)

    my_dpi = 96
    CONTAIN_OBJ = False
    for cls_ind, class_name in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = all_boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = all_scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
            continue
        colorVal = scalarMap.to_rgba(values[cls_ind])
        for i in inds:
            print(dets[i])
            bbox = dets[i, :4]
            score = dets[i, -1]
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor=colorVal, linewidth=3.5)
            )
            #ax.text(bbox[0], bbox[1] - 2,
            #        '{:s} {:.3f}'.format(class_name, score),
            #        bbox=dict(facecolor='blue', alpha=0.5),
            #        fontsize=14, color='white')

    ax.set_title(('object detections with '
                  'p(object | box) >= {:.1f}').format(CONF_THRESH), fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.show()
    if CONTAIN_OBJ:
        plt.axis('off')
        plt.tight_layout()
        directory = args.model_dir + 'prediction_whole'
        if not os.path.exists(directory):
            os.makedirs(directory)
        #print('printing to {}'.format(directory + '/' + 'pred_' + _name))
        # plt.figure(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
        plt.savefig(directory + '/' + 'pred_' + _name + '.jpg')
        plt.close()
    plt.close('all')


def save_predictions(all_boxes, all_scores):
    # Visualize detections for each class

    #image = image[:, :, (2, 1, 0)]

    # _, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(im, aspect='equal')

    #plt.imshow(image)

    #ax = plt.gca()
    # ax.set_autoscale_on(True)

    #my_dpi = 96
    #CONTAIN_OBJ = False
    for cls_ind, class_name in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = all_boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = all_scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
            continue
        #colorVal = scalarMap.to_rgba(values[cls_ind])
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            outline = _name + ' ' + str(score) + ' ' + str(np.round(bbox[0])) + ' ' + str(np.round(bbox[1])) + ' ' + str(
                np.round(bbox[2])) + ' ' + str(np.round(bbox[3]))
            # print(dets_final[i])
            # print(outline)
            filedict[class_name].write(outline + '\n')

CONF_THRESH = 0.7
NMS_THRESH = 0.3
# best values found so far: tennis-court,
# Notice: EPS should be maximum 0.99: there is a bug hindering using 1.0
#classTable = {'large-vehicle':EPS, 'swimming-pool':0.5, 'helicopter':EPS, 'bridge':0.5, 'plane':0.5,
#              'ship':EPS, 'soccer-ball-field':0.4, 'basketball-court':0.4,'ground-track-field':0.4,
#              'small-vehicle':EPS, 'harbor':EPS, 'baseball-diamond':0.4,'tennis-court':0.4, 'roundabout':0.5,
#              'storage-tank':0.5}

# using eps as 0.7 as reported by original paper
classTable = {'large-vehicle':EPS, 'swimming-pool':EPS, 'helicopter':EPS, 'bridge':EPS, 'plane':EPS,
              'ship':EPS, 'soccer-ball-field':EPS, 'basketball-court':EPS,'ground-track-field':EPS,
              'small-vehicle':EPS, 'harbor':EPS, 'baseball-diamond':EPS,'tennis-court':EPS, 'roundabout':EPS,
              'storage-tank':EPS}
read_format = 'txt'
save_format = 'xml'

cfg.TEST.HAS_RPN = True  # Use RPN for proposals
CONTAIN_OBJ = True

args = parse_args()

prototxt = os.path.join('models/pascal_voc', NETS[args.demo_net][0], 'faster_rcnn_end2end',
                        'test.prototxt')
caffemodel = os.path.join(args.model_dir, NETS[args.demo_net][1])

if not os.path.isfile(caffemodel):
    raise IOError(('{:s} not found.\nDid you run ./data/script/'
                   'fetch_faster_rcnn_models.sh?').format(caffemodel))

if args.cpu_mode:
    caffe.set_mode_cpu()
else:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    cfg.GPU_ID = args.gpu_id
net = caffe.Net(prototxt, caffemodel, caffe.TEST)

print('\n\nLoaded network {:s}'.format(caffemodel))

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np

NCURVES = 16
values = range(NCURVES)
jet = cm = plt.get_cmap('jet')
cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)


wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']


dstpath = '/home/azim_se/MyProjects/ECCV18/py-faster-rcnn/data/VOCdevkit/results/VOC2007/Task2_out'
#dstname = os.path.join(dstpath, name + '.txt')
filedict = {}
for cls in wordname_15:
    fd = open(os.path.join(dstpath, 'Task2_') + cls + '.txt', 'w')
    filedict[cls] = fd

for _mode in mode:
    params = mode[_mode]
    print("mode is {}".format(_mode))
    print('INT DIR is {}'.format(INT_DIR))
    print('OUT DIR is {}'.format(OUT_DIR))
    images_name = np.sort([name for name in os.listdir(INT_DIR + '/' + _mode + '/' + data[0]) if
                   name.endswith(('.png', '.jpg', '.jpeg', '.JPG'))])

    img_save_path = OUT_DIR + '/' + _mode + '/' + data[0]
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)


    print("reading names of images is done")
    import pdb

    for name in images_name:
        name_counter = 0
        scores_ = []
        # scores = []
        Scores = None
        boxes_ = []
        # boxes = []
        boxes = None
        _name = os.path.splitext(name)[0]  # basename(name)  removing extension

        print(colored('reading image:' + name, 'green'))
        # image = mpimage.imread(INT_DIR + '/' + _mode + '/' + data[0] + '/' + name)
        # image = Image.open(INT_DIR + '/' + _mode + '/' + data[0] + name)
        image = cv2.imread(INT_DIR + '/' + _mode + '/' + data[0] + '/' + name)
        print(colored('reading is done...', 'green'))


        image_size = np.shape(image)
        height = image_size[0]
        width = image_size[1]
        print('original image size is {}'.format(image_size))

        print(colored('cropping ' + _mode + ' dataset started...', 'green'))
        cropped = magic_test(image, mode[_mode], width, height)
        print('done.')
        count = 0
        for im, x, y in cropped:
            print(np.shape(im))
            scores_, boxes_ = demoCombined(net, im)

            #nms_vis_patch(boxes_, scores_, im, x, y, count)
            count += 1

            # pdb.set_trace()
            boxes_[:, list(range(0, 64, 2))] += x
            boxes_[:, list(range(1, 64, 2))] += y
            # pdb.set_trace()

            if boxes is None:
                boxes = np.copy(boxes_)
                Scores = np.copy(scores_)
            else:
                boxes = np.vstack((boxes, boxes_))
                Scores = np.vstack((Scores, scores_))

        print('image detection is done')
        print(len(Scores))
        print(len(boxes))
        print(len(Scores[0]))
        print(len(boxes[0]))




        # save_predictions(boxes, Scores)
        ## nms_vis_all(np.array(boxes), np.array(scores), image)
        nms_vis_all(boxes, Scores, image)
        plt.close('all')

    print(colored(name, 'green'), 'done.')
print(colored('----------------------', 'blue'))
