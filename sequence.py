from torch.utils.data import Dataset
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
import random
import re
import pandas as pd

class SequenceDataset(Dataset):
    def __init__(self,  mode='train', begin=0):
        super(SequenceDataset, self).__init__()
        file_path = 'train.csv'
        data = pd.read_csv(file_path)
        data = data['Sequence']
        if mode == 'val':
            random.shuffle(data)
            data = data[:len(data) // 10]
        elif mode == 'train':
            random.shuffle(data)
            data = data[:len(data)]
        data = [line+';' for line in data]
        self.data = data
        self.begin = begin
        self.count = len(self.data)

    def __getitem__(self, i):
        assert i < self.count and i >= 0, "Out of bounds"
        return self.data[i]

    def __len__(self):
        return self.count