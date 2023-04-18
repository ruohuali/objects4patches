import torch, torchvision
from torch import nn
from torchvision.io import read_image
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image

from models import ViTBackbone
from utils import getVisualizableTransformedImageFromPIL, HWC2CHW


if __name__ == '__main__':
    model = ViTBackbone(pretrained=True)
    vit, weights = model.vit, model.vit_weights
    image_paths = ['example_images/cat.jpg', 'example_images/dog.jpg', 'example_images/car.jpg']
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image = getVisualizableTransformedImageFromPIL(image, model.vit_weights.transforms())
        image = HWC2CHW(image)
        images.append(image)

    model.inference(images.copy())

    y = model(images.copy(), feature_extraction=True, cls_feature=True)
    