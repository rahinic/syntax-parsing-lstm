import torch
from torch.utils.data import DataLoader
from datasetConLL2003 import SlidingWindowDataset
validation_dataset = DataLoader(dataset=SlidingWindowDataset(),
                                batch_size=64,
                                shuffle=False)



for idx, (sample, label) in enumerate(validation_dataset):

    print(idx)
    print(sample)
    print(label)
                          