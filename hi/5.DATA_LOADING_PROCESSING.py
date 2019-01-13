# 数据下载地址https://download.pytorch.org/tutorial/faces.zip
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':

    plt.ion()

    landmarks_frame = pd.read_csv('../data/faces/face_landmarks.csv')
    # print(landmarks_frame)

    n = 65  # 选第65个图片
    img_name = landmarks_frame.iloc[n, 0]
    print(img_name)
    landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
    landmarks = landmarks.astype('float').reshape(-1, 2)

    print('image name : {}'.format(img_name))
    print('landmark shape: {}'.format(landmarks.shape))
    print('First 4 landmarks : {}'.format(landmarks[:4]))
