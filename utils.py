import torch, torchvision
from torchvision.io import read_image
import torch.nn.functional as F
from models import ViTBackbone, preprocess

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def cosineSimilarity(features, temperature=0.07, softmax=False):
    features = F.normalize(features, dim=2)
    similarity_matrix = torch.matmul(features, features.permute(0, 2, 1)) / temperature
    if softmax:
        similarity_matrix = F.softmax(similarity_matrix, dim=1)
    return similarity_matrix

def visualizeCentercrop(transforms):
    from PIL import Image
    from torchvision.transforms import CenterCrop

    # Load an image using PIL
    image_path = 'car.jpg'
    image = Image.open(image_path)

    # Define the desired center crop size
    crop_size = 224

    # # Create a CenterCrop transform
    # center_crop_transform = CenterCrop(crop_size)

    # # Apply the center crop transform to the image
    # center_cropped_image = center_crop_transform(image)

    center_cropped_image = transforms(image)
    center_cropped_image = center_cropped_image.permute(1, 2, 0)

    # # Display the original and center-cropped images
    # import matplotlib.pyplot as plt

    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # axes[0].imshow(image)
    # axes[0].set_title('Original Image')
    # axes[1].imshow(center_cropped_image)
    # axes[1].set_title(f'Center Cropped Image ({crop_size}x{crop_size})')
    # plt.show()    
    return center_cropped_image

def visualizeSimilarity(image, similarity_matrix, patch_idx):
    image += abs(image.min())
    image /= image.max()
    similarity_matrix = similarity_matrix.squeeze()
    # patch_idx = 100
    patch_similarities = similarity_matrix[patch_idx].view(14, 14)
    patch_row_idx = patch_idx // 14
    patch_col_idx = patch_idx - patch_row_idx * 14
    print(patch_row_idx, patch_col_idx)
    patch_idx_plot = torch.zeros_like(patch_similarities)
    patch_idx_plot[patch_row_idx, patch_col_idx] = 0.99

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[1].imshow(patch_similarities)
    axes[1].set_title(f'sim')
    axes[2].imshow(patch_idx_plot)
    axes[2].set_title(f'sim')    
    plt.savefig(f'images/{patch_idx}.jpg') 


if __name__ == '__main__':
    model = ViTBackbone(pretrained=True)
    print(model.vit_weights.transforms())
    model.eval()
    vit, weights = model.vit, model.vit_weights
    image1 = read_image('car.jpg')
    batch = preprocess([image1], weights.transforms())

    features = model(batch, feature_extraction=True)
    similarity_matrix = cosineSimilarity(features, softmax=True, temperature=1)
    image = visualizeCentercrop(model.vit_weights.transforms())
    with torch.no_grad():
        for i in range(0, 180):
            visualizeSimilarity(image, similarity_matrix, i)