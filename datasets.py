import os
import h5py
import os.path as osp
import numpy as np
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data

__all__ = ['EPIDataSetTrain', 'EPIDataSetTest']


class EPIDataSetTrain(data.Dataset):
    def __init__(self, data_tr, label_tr):
        super(EPIDataSetTrain, self).__init__()
        self.data = data_tr
        self.label = label_tr

        assert len(self.data) == len(self.label), \
            "the number of sequences and labels must be consistent."

        print("The number of positive data is {}".format(sum(self.label.reshape(-1) == 1)))
        print("The number of negative data is {}".format(sum(self.label.reshape(-1) == 0)))
        print("pre-process data is done.")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_one = self.data[index]
        label_one = self.label[index]

        return {"data": data_one, "label": label_one}


class EPIDataSetTest(data.Dataset):
    def __init__(self, data_te, label_te):
        super(EPIDataSetTest, self).__init__()
        self.data = data_te
        self.label = label_te

        assert len(self.data) == len(self.label), \
            "the number of sequences and labels must be consistent."
        print("The number of positive data is {}".format(sum(self.label.reshape(-1) == 1)))
        print("The number of negative data is {}".format(sum(self.label.reshape(-1) == 0)))
        print("pre-process data is done.")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_one = self.data[index]
        label_one = self.label[index]

        return {"data": data_one, "label": label_one}


