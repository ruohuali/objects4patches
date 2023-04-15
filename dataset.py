import torch, torchvision
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils import getVisualizableTransformedImageFromPIL

def getVOCDataloader(path, batch_size, transform=None, download=False):
    dataset = torchvision.datasets.VOCDetection(path, year='2012', image_set='trainval', transform=transform, download=download)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    return dataloader

if __name__ == '__main__':
    dataloader = getVOCDataloader('..', 5)
    weights = ViT_B_16_Weights.DEFAULT
    print(len(dataloader))
    for i, x in enumerate(dataloader):
        for xx in x:
            image = xx[0]
            plt.figure()
            image = getVisualizableTransformedImageFromPIL(image, weights.transforms())
            plt.imshow(image)
            plt.show()
        if i == 3:
            break
