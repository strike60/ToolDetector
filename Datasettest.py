#!/usr/bin/python3
#-*- coding:UTF-8 -*-
import os
import numpy as np
import cv2
import sys
import argparse
import tensorflow as tf


_ImageWidth = 32
_ImageHeight = 32
_LabelBytes = 1
_ImageBytes = _ImageWidth*_ImageHeight*3

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str, default='./Dataset',
    help='Directory to images')


def ParseDataset(data_dir):
    imgs = []
    labels = []
    if not os.path.exists(data_dir):
        raise ValueError(data_dir+' is not existed')
    datalist = sorted(os.listdir(data_dir))
    for datasetpath in datalist:
        datasetpath = os.path.join(data_dir, datasetpath)
        label, img = ReadOneExample(datasetpath)
        imgs.append(img)
        labels.append(label)
    for i in range(15):
        print(labels[i])
        cv2.imshow('image' + str(i), np.reshape(imgs[i], (32, 32, 3)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def ReadOneExample(datasetpath):
    with open(datasetpath, 'rb') as f:
        label = f.read(_LabelBytes)
        label = np.fromstring(label, np.uint8)
        image = f.read(_ImageBytes)
        image = np.fromstring(image, dtype=np.uint8)
        return label, image
        # raw = tf.decode_raw(f.read(_LabelBytes+_ImageBytes), tf.uint8)
        # label = tf.cast(raw[0], tf.int32)
        # image = raw[_LabelBytes: _ImageBytes+1]
        # sess = tf.Session()
        # return sess.run([label, image])


def main():
    ParseDataset(FLAGS.data_dir)


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    main()
