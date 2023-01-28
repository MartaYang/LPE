import os.path as osp
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import clip


class CUB(Dataset):

    def __init__(self, setname, args=None, preprocess=None, return_path=False):
        IMAGE_PATH = os.path.join(args.data_dir, 'cub/')
        SPLIT_PATH = os.path.join(args.data_dir, 'cub/split/')
        txt_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1
        self.wnids = []
        self.args = args
        if setname == 'train':
            if len(lines)>5864:
                lines.pop(5864)  #this image file is broken
        if args.semantic_path is not None:
            self.label2vec = []
            self.wnid2vec = np.load(args.semantic_path, allow_pickle=True).item()

        for l in lines:
            context = l.split(',')
            name = context[0]
            wnid = context[1]
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                if args.semantic_path is not None:
                    self.label2vec.append(self.wnid2vec[wnid.split('.')[-1]])
                    # self.label2name.append(wnid2name[wnid])
                lb += 1

            data.append(path)
            label.append(lb)
        if args.semantic_path is not None:
            self.label2vec = nn.Parameter(torch.from_numpy(np.array(self.label2vec)).float(), requires_grad=False).cuda()
        else:
            self.label2vec = None

        self.data = data
        self.label = label
        self.num_class = np.unique(np.array(label)).shape[0]
        self.return_path = return_path

        if setname == 'train':

            image_size = 84

            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        else:

            image_size = 84
            resize_size = 92

            self.transform = transforms.Compose([
                transforms.Resize([resize_size, resize_size]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        if self.return_path:
            return image, label, path
        else:
            return image, label


if __name__ == '__main__':
    pass
