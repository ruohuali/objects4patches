import torch, torchvision
from torchvision.io import read_image
import torch.nn.functional as F
from models import ViTBackbone, preprocess

def cosSimilarity(features):
    similiarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(2), dim=3)
    print('shape', features.size(), similiarity_matrix.size())

if __name__ == '__main__':
    model = ViTBackbone(pretrained=True)
    vit, weights = model.vit, model.vit_weights
    image1 = read_image('dog.jpg')
    image2 = read_image('cat.jpg')
    image3 = read_image('car.jpg')    
    batch = preprocess([image1, image2, image3], weights.transforms())

    features = model(batch, feature_extraction=True)
    generateSimilarityMatrix(features)