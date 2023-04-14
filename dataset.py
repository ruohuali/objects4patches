import torch, torchvision
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from utils import recoverTransformedImage

def getVOCDataset(path, transform, download=False):
    dataset = torchvision.datasets.VOCDetection(path, year='2012', image_set='trainval', transform=transform, download=download)
    return dataset

if __name__ == '__main__':
    weights = ViT_B_16_Weights.DEFAULT
    dataset = getVOCDataset('..', weights.transforms())
    print(len(dataset))
    for i in range(10):
        x = dataset[i]
        print(type(x))
        print(x[0].shape)
        print(x[1])
        plt.figure()
        image = recoverTransformedImage(x[0])
        plt.imshow(image)
        plt.show()
