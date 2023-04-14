import torch, torchvision

path = '..'
dataset = torchvision.datasets.VOCDetection(path, year='2012', image_set='train', download=True)
print(len(dataset))