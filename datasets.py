import torch, torchvision
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from utils import getVisualizableTransformedImageFromPIL, getVisualizableTransformedImageFromTensor

def getVOCDataloader(path, batch_size, ratio=1, train_ratio=0.9, shuffle=True, transform=None, download=False):
    dataset = torchvision.datasets.VOCDetection(path, year='2012', image_set='trainval', transform=transform, download=download)

    idx = int(len(dataset) * ratio)
    split = range(0, idx)
    dataset = Subset(dataset, split)

    train_idx = int(len(dataset) * train_ratio)
    train_split = range(0, train_idx)
    train_dataset = Subset(dataset, train_split)
    validate_split = range(train_idx, len(dataset))
    validate_dataset = Subset(dataset, validate_split)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: x)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    return train_loader, validate_loader

def getCaltechDataloader(path, batch_size, ratio=1, train_ratio=0.9, shuffle=True, transform=None, download=False):
    dataset = torchvision.datasets.Caltech101(path, transform=transform, download=download)

    idx = int(len(dataset) * ratio)
    split = range(0, idx)
    dataset = Subset(dataset, split)

    train_idx = int(len(dataset) * train_ratio)
    train_split = range(0, train_idx)
    train_dataset = Subset(dataset, train_split)
    validate_split = range(train_idx, len(dataset))
    validate_dataset = Subset(dataset, validate_split)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: x)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    return train_loader, validate_loader

if __name__ == '__main__':
    transforms = transforms.Compose([transforms.CenterCrop])
    train_loader, validate_loader = getVOCDataloader('..', 5)
    weights = ViT_B_16_Weights.DEFAULT
    print(len(train_loader), len(validate_loader))
    for i, x in enumerate(train_loader):
        print('what', len(x))
        for xx in x:
            print(xx)
            image = xx[0]
            plt.figure()
            image = getVisualizableTransformedImageFromTensor(image, weights.transforms())
            plt.imshow(image)
            plt.show()
        if i == 0:
            break

    # ====================================================================
    
    train_loader, validate_loader = getCaltechDataloader('../', 3, ratio=0.01, download=False)
    for i, x in enumerate(train_loader):
        print(len(x), type(x[0]), type(x[0][0]), type(x[0][1]))

        break
