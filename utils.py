import torch, torchvision
from torchvision.io import read_image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from models import ViTBackbone, preprocess
from PIL import Image
import matplotlib.pyplot as plt
import io

from losses import cosineSimilarity

def CHW2HWC(image):
    return image.permute(1, 2, 0)

def HWC2CHW(image):
    return image.permute(2, 0, 1)

def recoverTransformedImage(transformed_image):
    image = CHW2HWC(transformed_image)
    image += abs(image.min())
    image /= image.max()    
    image = (image * 255).to(torch.uint8)
    return image

def getVisualizableTransformedImageFromPIL(image, transforms):
    image = image.convert('RGB')
    image = TF.to_tensor(image)
    image = getVisualizableTransformedImageFromTensor(image, transforms)
    return image

def getVisualizableTransformedImageFromTensor(image, transforms):
    image = transforms(image)
    image = recoverTransformedImage(image)
    return image

def visualizePatchSimilarities(image, similarity_matrix, patch_idx, save=True):
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
    if save:
        plt.savefig(f'images/{patch_idx}.jpg') 

    plot_buffer = io.BytesIO()
    plt.savefig(plot_buffer, format='png')
    plt.close()
    plot_image = Image.open(plot_buffer)
    plot_tensor = TF.to_tensor(plot_image)  
    return plot_tensor

def visualizeLabels(images, labels, features_shape):
    B, P2, D = features_shape
    row_num = int(P2**(1 / 2))
    labels = labels.view(B, P2)
    for image, label in zip(images, labels):
        label = label.view(row_num, row_num)
        image = CHW2HWC(image)
        print(f'class id {label.unique()}')

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image)
        axes[0].set_title('image')
        axes[1].imshow(label)
        axes[1].set_title(f'label')
        plt.show()

if __name__ == '__main__':
    model = ViTBackbone(pretrained=True)
    print(model.vit_weights.transforms())
    model.eval()
    vit, weights = model.vit, model.vit_weights
    image_path = 'example_images/cat.jpg'
    image1 = Image.open(image_path)
    images = [image1]

    features = model(images, feature_extraction=True)
    features = F.normalize(features, dim=-1)
    print(features.shape)
    similarity_matrix = cosineSimilarity(features.squeeze(), softmax=True, temperature=1)
    image = getVisualizableTransformedImageFromPIL(Image.open(image_path), model.vit_weights.transforms())
    with torch.no_grad():
        for i in range(180):
            print(i)
            visualizePatchSimilarities(image, similarity_matrix, i)
    #     x = visualizePatchSimilarities(image, similarity_matrix, 50, save=False)
    # plt.figure()
    # x = CHW2HWC(x)
    # plt.imshow(x)
    # plt.show()
