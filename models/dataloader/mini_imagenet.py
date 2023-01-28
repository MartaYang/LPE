import os.path as osp
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import clip
from skimage.util import img_as_float
from skimage.filters import sobel
from skimage.color import rgb2gray
from skimage.segmentation import watershed

class MiniImageNet(Dataset):

    def __init__(self, setname, args, preprocess=None, return_path=False):
        IMAGE_PATH = os.path.join(args.data_dir, 'miniimagenet/images')
        SPLIT_PATH = os.path.join(args.data_dir, 'miniimagenet/split')

        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []
        self.args = args
        if args.semantic_path is not None:
            self.label2vec = []
            self.wnid2vec = np.load(args.semantic_path, allow_pickle=True).item()

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                if args.semantic_path is not None:
                    self.label2vec.append(self.wnid2vec[wnid])
                lb += 1
            data.append(path)
            label.append(lb)
        if args.semantic_path is not None:
            self.label2vec = nn.Parameter(torch.from_numpy(np.array(self.label2vec)).float(), requires_grad=False).cuda()
        else:
            self.label2vec = None

        self.data = data  # data path of all data
        self.label = label  # label of all data
        self.num_class = len(set(label))
        self.preprocess = preprocess
        self.return_path = return_path

        if setname == 'val' or setname == 'test':

            image_size = 84
            resize_size = 92

            self.transform = transforms.Compose([
                transforms.Resize([resize_size, resize_size]),
                transforms.CenterCrop(image_size),

                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif setname == 'train':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        img = Image.open(path).convert('RGB')
        image = self.transform(img)
        if self.preprocess != None:
            image_clip = self.preprocess(img)
            return image, image_clip, label
        else:
            return image, label




if __name__ == '__main__':
    pass
