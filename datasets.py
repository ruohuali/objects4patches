import torch, torchvision
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader, Subset, random_split
import matplotlib.pyplot as plt
import numpy as np
import random

from utils import getVisualizableTransformedImageFromPIL, getVisualizableTransformedImageFromTensor

def randomSample(dataset, ratio):
    # Get the total size of the dataset
    dataset_size = len(dataset)
    
    # Create a list of indices for the dataset
    indices = list(range(dataset_size))
    
    # Shuffle the indices
    random.shuffle(indices)
    
    # Select a subset of the shuffled indices
    subset_indices = indices[:len(dataset) * ratio]
    
    # Use the Subset class from PyTorch to create a new dataset with the subset indices
    subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
    
    return subset_dataset

def getVOCDataloader(path, batch_size, ratio=1, ratios=(0.9, 0.1), shuffle=True, transform=None, download=False):
    dataset = torchvision.datasets.VOCDetection(path, year='2012', image_set='trainval', transform=transform, download=download)
    dataset = Subset(dataset, range(int(len(dataset) * ratio)))

    train_dataset, validate_dataset = random_split(dataset, ratios)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: x)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    return train_dataloader, validate_dataloader

def getCaltechDataloader(path, batch_size, ratio=1, ratios=(0.8, 0.1, 0.1), shuffle=True, transform=None, download=False):
    dataset = torchvision.datasets.Caltech101(path, transform=transform, download=download)
    dataset = Subset(dataset, range(int(len(dataset) * ratio)))

    train_dataset, validate_dataset, test_dataset = random_split(dataset, ratios)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: x)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)    

    return train_dataloader, validate_dataloader, test_dataloader

if __name__ == '__main__':
    transforms = transforms.Compose([transforms.CenterCrop])
    train_dataloader, validate_dataloader = getVOCDataloader('..', 5)
    weights = ViT_B_16_Weights.DEFAULT
    # print(len(train_dataloader), len(validate_dataloader))
    # for i, x in enumerate(train_dataloader):
    #     print('what', len(x))
    #     for xx in x:
    #         print(xx)
    #         image = xx[0]
    #         plt.figure()
    #         plt.title('first')
    #         image = getVisualizableTransformedImageFromTensor(image, weights.transforms())
    #         plt.imshow(image)
    #         plt.show()
    #     if i == 0:
    #         break

    # ====================================================================
    
    train_dataloader, validate_dataloader, test_dataset = getCaltechDataloader('../', 3, ratio=0.01, download=False)
    for i, x in enumerate(train_dataloader):
        print(len(x), type(x[0]), type(x[0][0]), type(x[0][1]))
        for xx in x:
            print(xx)
            image = xx[0]
            print(xx[1])
            plt.figure()
            plt.title('second')
            image = getVisualizableTransformedImageFromTensor(image, weights.transforms())
            plt.imshow(image)
            # plt.show()
        if i == 0:
            break
