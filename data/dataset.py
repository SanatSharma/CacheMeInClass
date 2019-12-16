import numpy as np
from imutils import paths
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os
from utils import *


class Example:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return str(self.x) + ":" + str(self.y)

    def __repr__(self):
        return str(self)

class FaceDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.student_indexer = Indexer()
        self.image_indexer = Indexer()
        self.examples = self.index_images(self.data_path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        sample= {'path': self.image_indexer.get_object(self.examples[idx].x), 'label':self.examples[idx].y}
        return sample
    
    def index_images(self, data_path):
        image_paths = list(paths.list_images(data_path))
        examples = [Example(self.image_indexer.get_index(image_path),
            self.student_indexer.get_index(image_path.split(os.path.sep)[-2]))
            for image_path in image_paths]
        print(examples)
        return examples

def create_dataset(data_path, args, cutoff=.8, shuffle=True):
    dataset = FaceDataset(data_path)
    print("Dataset size:", len(dataset))

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(cutoff * dataset_size)
    if shuffle :
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    #print(len(train_sampler), len(valid_sampler))

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                            sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.test_batch_size,
                                                sampler=valid_sampler)
    return train_loader, test_loader, dataset.student_indexer