from skimage import transform, io
from torch.utils.data import Dataset
import os


class GsDataset(Dataset):
    """为我们的数据集单独定制一个类，继承torch里的Dataset类"""
    def __init__(self, root_dir, trans=None):
        """
        :param root_dir(String):  Directory with all the images.
        :param trans(callable, optional):  Optional transform to be applied on a sample.
        """
        self.transform = trans
        self.images = ergodic_all_pic(root_dir)

    def __getitem__(self, item):
        image_name = self.images[item]
        sample = io.imread(image_name)
        return sample

    def __len__(self):
        return len(self.images)


class Resize(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample
        if isinstance(self.output_size, int):
            size = (self.output_size, self.output_size)
        else:
            size = self.output_size
        new_image = transform.resize(image, size)
        return new_image


def ergodic_dir(path, r_full=True):
    s = path[-1]
    if s == '/':
        path = path[:-1]
    # print(path)
    path_dir = os.listdir(path)
    if not r_full:
        return path_dir
    di_fi = []
    for di_or_fi in path_dir:
        each_path = os.path.join('%s/%s' % (path, di_or_fi))
        # print(each_path)
        di_fi.append(each_path)
    return di_fi


def ergodic_all_pic(path):
    """想象一个树结构，以path为根节点，文件为叶子结点，返回所有叶子结点的路径"""
    li_tmp = []
    li_fi = []
    li_suffix = ['jpg', 'jpeg', 'png', 'bmp']
    for i in ergodic_dir(path):
        if os.path.isdir(i):
            li_tmp += ergodic_all_pic(i)
        else:
            if i.split('.')[-1] in li_suffix:
                li_fi.append(i)

    return li_tmp + li_fi


if __name__ == '__main__':
    import time
    import warnings
    warnings.filterwarnings('ignore')

    # os.environ['CUDA_VISIBLE_DEVICES'] = "4"
    print('-' * 15, 'Start', time.ctime(), '-' * 15, '\n')

    dir_path = 'E:/图片/Saved Pictures/'
    img_dataset = GsDataset(dir_path, trans=Resize(224))

    pic_all = ergodic_all_pic(dir_path)
    print(len(pic_all))
    # num_imgs = len(img_dataset)
    # print(num_imgs)
    # for i in range(num_imgs):
    #     each_sample = img_dataset[i]
    #     print(each_sample.shape)

    print('%s%s %s %s %s' % ('\n', '-' * 16, 'End', time.ctime(), '-' * 16))

