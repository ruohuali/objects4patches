import torch, torchvision
from torchvision.io import read_image
import torch.nn.functional as F
from models import ViTBackbone, preprocess

from PIL import Image
import matplotlib.pyplot as plt

def cosineSimilarity(features, temperature=0.07, softmax=False):
    features = F.normalize(features, dim=2)
    similarity_matrix = torch.matmul(features, features.permute(0, 2, 1)) / temperature
    if softmax:
        similarity_matrix = F.softmax(similarity_matrix, dim=1)
    return similarity_matrix

def getVisualizableTransformedImage(image_path, transforms):
    image = Image.open(image_path)
    image = transforms(image)
    image = image.permute(1, 2, 0)
    image += abs(image.min())
    image /= image.max()    
    return image

def visualizePatchSimilarities(image, similarity_matrix, patch_idx):
    similarity_matrix = similarity_matrix.squeeze()
    assert similarity_matrix.dim() == 2 and similarity_matrix.size(0) == similarity_matrix.size(1)
    row_num = int(similarity_matrix.size(0)**(1 / 2))
    patch_similarities = similarity_matrix[patch_idx].view(row_num, row_num)
    patch_row_idx = patch_idx // row_num
    patch_col_idx = patch_idx - patch_row_idx * row_num
    patch_idx_plot = torch.zeros_like(patch_similarities)
    patch_idx_plot[patch_row_idx, patch_col_idx] = 1

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title('image')
    axes[1].imshow(patch_similarities)
    axes[1].set_title(f'patch_similarities')
    axes[2].imshow(patch_idx_plot)
    axes[2].set_title(f'patch_idx_plot')    
    plt.savefig(f'images/{patch_idx}.jpg') 
    # plt.show()

if __name__ == '__main__':
    model = ViTBackbone(pretrained=True)
    print(model.vit_weights.transforms())
    model.eval()
    vit, weights = model.vit, model.vit_weights
    image_path = 'dog.jpg'
    image1 = read_image(image_path)
    batch = preprocess([image1], weights.transforms())

    features = model(batch, feature_extraction=True)
    similarity_matrix = cosineSimilarity(features, softmax=True, temperature=1)
    image = getVisualizableTransformedImage(image_path, model.vit_weights.transforms())
    with torch.no_grad():
        for i in range(180):
            print(i)
            visualizePatchSimilarities(image, similarity_matrix, i)