import os.path as osp
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np


class DatasetLoader(Dataset):

    def __init__(self, setname, args, preprocess=None, return_path=False):

        DATASET_DIR = os.path.join(args.data_dir, 'cifar_fs')

        # Set the path according to train, val and test
        if setname == 'train':
            THE_PATH = osp.join(DATASET_DIR, 'meta-train')
            label_list = os.listdir(THE_PATH)
        elif setname == 'test':
            THE_PATH = osp.join(DATASET_DIR, 'meta-test')
            label_list = os.listdir(THE_PATH)
        elif setname == 'val':
            THE_PATH = osp.join(DATASET_DIR, 'meta-val')
            label_list = os.listdir(THE_PATH)
        else:
            raise ValueError('Unkown setname.')

        label_list.sort()
        data = []
        label = []

        if args.semantic_path is not None:
            self.label2vec = []
            self.name2vec = np.load(args.semantic_path, allow_pickle=True).item()

        folders = [osp.join(THE_PATH, label) for label in label_list if os.path.isdir(osp.join(THE_PATH, label))]

        # Get the images' paths and labels
        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            this_folder_images.sort()
            name = this_folder.split('/')[-1]
            # print(this_folder, name)
            self.label2vec.append(self.name2vec[name])
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)
        if args.semantic_path is not None:
            self.label2vec = nn.Parameter(torch.from_numpy(np.array(self.label2vec)).float(), requires_grad=False).cuda()
            print(self.label2vec.shape)
        else:
            self.label2vec = None

        self.data = data
        self.label = label
        self.num_class = len(set(label))
        self.return_path = return_path

        # Transformation
        if setname == 'train':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))])

        else:

            image_size = 84
            resize_size = 92

            self.transform = transforms.Compose([
                transforms.Resize([resize_size, resize_size]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))])

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

