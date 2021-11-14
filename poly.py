from torch.utils.data import Dataset
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence

class PolynomialDataset(Dataset):
    def __init__(self,  begin=0):
        super(PolynomialDataset, self).__init__()
        file_path = 'train.txt'
        data = open(file_path, 'r').readlines()
        data = [line.strip().split("=") for line in data]
        self.data = data
        self.begin = begin
        self.count = len(self.data)
        
    def __getitem__(self, i):
        assert i < self.count and i >= 0, "Out of bounds"
        return self.data[i]

    def __len__(self):
        return self.count