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


def show_landmarks(img, landmark):
    # plt.figure()
    plt.imshow(img)
    plt.scatter(landmark[:, 0], landmark[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)


# def get_picture_path(name):
#     now_path = os.getcwd()
#     pro_path = '/'.join(now_path.split('\\')[:-1])
#     return os.path.join(pro_path+'/data/faces/', name)


class FaceLandmarkDataset(Dataset):
    """为我们的数据集单独定制一个类，继承torch里的Dataset类"""
if __name__ == '__main__':

    plt.ion()

    landmarks_frame = pd.read_csv('../data/faces/face_landmarks.csv')
    # print(landmarks_frame)

    n = 65  # 选第65个图片
    img_name = landmarks_frame.iloc[n, 0]
    # print(img_name)
    landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
    landmarks = landmarks.astype('float').reshape(-1, 2)

    print('image name : {}'.format(img_name))
    print('landmark shape: {}'.format(landmarks.shape))
    print('First 4 landmarks : {}'.format(landmarks[:4]))

    img_path = os.path.join('../data/faces/', img_name)
    image1 = io.imread(img_path)
    # print(image1)
    show_landmarks(image1, landmarks)
    # io.imshow(image1)
