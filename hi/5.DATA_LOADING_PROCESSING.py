# 第五节 数据加载和预处理
# 案例数据下载地址https://download.pytorch.org/tutorial/faces.zip
from __future__ import print_function, division
import os
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
from torchvision.transforms import Compose
import warnings
warnings.filterwarnings('ignore')


def show_landmarks(img, landmark):
    # plt.figure()
    plt.imshow(img)
    plt.scatter(landmark[:, 0], landmark[:, 1], s=10, marker='.', c='r')
    # plt.pause(0.001)  # 这一句不注释就会导致后面使用subplot时，各个subplot会单独占一整个图


# def get_picture_path(name):
#     now_path = os.getcwd()
#     pro_path = '/'.join(now_path.split('\\')[:-1])
#     return os.path.join(pro_path+'/data/faces/', name)


class FaceLandmarkDataset(Dataset):
    """为我们的数据集单独定制一个类，继承torch里的Dataset类"""
    def __init__(self, scv_dir, root_dir, trans=None):
        """

        :param scv_dir(String):   Path to the csv file with annotations.
        :param root_dir(String):  Directory with all the images.
        :param trans(callable, optional):  Optional transform to be applied on a sample.
        """
        self.landmarks_frame = pd.read_csv(scv_dir)
        self.root_dir = root_dir
        self.transform = trans

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, item):
        image_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[item, 0])
        image = io.imread(image_name)
        landmark = self.landmarks_frame.iloc[item, 1:].as_matrix()
        landmark = landmark.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmark}

        if self.transform:
            sample = self.transform(sample)

        return sample


'''We will write them as callable classes instead of simple functions so that parameters of the transform 
need not be passed every time it’s called. '''


class Rescale(object):
    """我觉得就是resize，不知道为什么这里要叫rescale"""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmark = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size*h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size*w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)  # 这样好吗
        new_image = transform.resize(image, (new_h, new_w))
        new_landmark = landmark * [new_w/w, new_h/h]  # 这里也会默认取整，不太好吧

        return {'image': new_image, 'landmarks': new_landmark}


class RandomCrop(object):
    """随机裁剪图片，变成一个指定的尺寸"""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmark = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top, left = np.random.randint(0, h-new_h), np.random.randint(0, w-new_w)
        new_image = image[top: top+new_h, left: left+new_w]
        new_landmark = landmark - [left, top]
        return {'image': new_image, 'landmarks': new_landmark}


class ToTensor(object):
    """把numpy数组转成tensor"""
    def __call__(self, sample):
        image, landmark = sample['image'], sample['landmarks']
        image = image.transpose((2, 0, 1))
        return {'image': image, 'landmarks': landmark}


'''One parameter of interest is collate_fn. You can specify how exactly the samples need to be batched using collate_fn.
 However, default collate should work fine for most use cases.'''


def show_landmark_by_batch(sample_batch):
    img, ldm = sample_batch['image'], sample_batch['landmarks']
    batch_size = len(sample_batch)
    im_size = img.size(2)
    grid = utils.make_grid(img)  # 一个做格子的函数，把好几张图按顺序排好队，做成对应的格子
    plt.imshow(grid.numpy().transpose(1, 2, 0))

    for z in range(batch_size):
        plt.scatter(ldm[z, :, 0].numpy() + z*im_size,
                    ldm[z, :, 1].numpy(),
                    s=10, marker='.', c='r')
        plt.title('Batch from dataLoader')


if __name__ == '__main__':

    plt.ion()  # 打开交互模式

    # landmarks_frame = pd.read_csv('../data/faces/face_landmarks.csv')
    # # print(landmarks_frame)
    #
    # n = 65  # 选第65个图片
    # img_name = landmarks_frame.iloc[n, 0]
    # # print(img_name)
    # landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
    # landmarks = landmarks.astype('float').reshape(-1, 2)
    #
    # print('image name : {}'.format(img_name))
    # print('landmark shape: {}'.format(landmarks.shape))
    # print('First 4 landmarks : {}'.format(landmarks[:4]))
    #
    # img_path = os.path.join('../data/faces/', img_name)
    # image1 = io.imread(img_path)
    # # print(image1)
    # show_landmarks(image1, landmarks)
    # # io.imshow(image1)

    landmark_path = '../data/faces/face_landmarks.csv'
    dataset_path = '../data/faces/'

    face_dataset = FaceLandmarkDataset(landmark_path, dataset_path)

    fig = plt.figure()

    # # 分别看所有图的标记
    # for i in range(len(face_dataset)):
    #     each_sample = face_dataset[i]
    #     show_landmarks(each_sample['image'], each_sample['landmarks'])

    for i in range(len(face_dataset)):  # 看前四张图的标记
        each_sample = face_dataset[i]
        # print(each_sample)
        # print(i, each_sample['image'].shape, each_sample['landmarks'].shape)
        sub_p = plt.subplot(1, 4, i+1)
        sub_p.set_title('Sample #{}'.format(i+1))
        plt.axis('off')  # 取消坐标轴
        show_landmarks(each_sample['image'], each_sample['landmarks'])
        if i == 3:
            break
    plt.tight_layout()  # 会自动调整子图参数，使之填充整个图像区域
    plt.show()

    # 定义了上面的rescale, randomCrop, ToTensor三个类之后，就能使用torch包里的一个组合函数了
    scale = Rescale(256)
    crop = RandomCrop(128)
    composed = Compose([Rescale(256), RandomCrop(224)])  # 组合使用两个功能

    fig = plt.figure()
    plt.pause(0.1)
    sample1 = face_dataset[65]
    for i, ts in enumerate([scale, crop, composed]):  # 展示各个功能对图片尺寸的修改
        transformed = ts(sample1)

        sub_p = plt.subplot(1, 3, i+1)
        plt.tight_layout()
        sub_p.set_title(type(ts).__name__)
        show_landmarks(transformed['image'], transformed['landmarks'])
    plt.show()

    transformed_data = FaceLandmarkDataset(landmark_path, dataset_path,
                                           Compose([Rescale(256), RandomCrop(224), ToTensor()]))
    for i in range(len(transformed_data)):
        each_sample = transformed_data[i]
        print(i, each_sample['image'].shape, each_sample['landmarks'].shape)
        if i == 3:
            break

    data_loader = DataLoader(transformed_data, batch_size=4, shuffle=True, num_workers=4)
    for i, batch_sample in enumerate(data_loader):
        print(i, batch_sample['image'].shape, batch_sample['landmarks'].shape)
        if i == 3:
            plt.figure()
            # show_landmark_by_batch(batch_sample)
            show_landmark_by_batch(batch_sample)
            plt.axis('off')
            plt.show()
            break

    plt.ioff()  # 关闭交互模式

    # # 复制粘贴。因为下面这个暂时不能用，没有数据
    # import torch
    # from torchvision import transforms, datasets
    #
    # data_transform = transforms.Compose([
    #     transforms.RandomSizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])
    # hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
    #                                            transform=data_transform)
    # dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
    #                                              batch_size=4, shuffle=True,
    #                                              num_workers=4)
