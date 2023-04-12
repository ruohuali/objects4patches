import torch, torchvision
from torch import nn
from torchvision.io import read_image
from torchvision.models import vit_b_16, ViT_B_16_Weights

def getViTModel(pretrained=False):
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights) if pretrained else vit_b_16()
    return model, weights

def preprocess(images, model_preprocess):
    batch = []
    for image in images:
        x = model_preprocess(image)
        batch.append(x)
    batch = torch.stack(batch)
    if batch.dim() < 4:
        batch = batch.unsqueeze(0)    
    return batch

def inference(model, batch, class_names, topk=3):
    model.eval()
    logits = model(batch, feature_extraction=False)
    predictions = logits.softmax(dim=1)
    for prediction in predictions:
        topk_ids = torch.topk(prediction, topk).indices
        for idx in topk_ids:
            score = prediction[idx].item()
            category_name = class_names[idx]
            print(f"{category_name}: {100 * score:.1f}%")
        print('-' * 50)

def experiment(model, batch):
    features = model(batch, feature_extraction=True)
    print('feature shape', features.size())

class ViTBackbone(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.vit, self.vit_weights = getViTModel(pretrained=pretrained)

    def forward(self, x, feature_extraction=True):
        # Reshape and permute the input tensor
        x = self.vit._process_input(x)
        n = x.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.vit.encoder(x)

        if feature_extraction:
            return x[:, 1:]

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        x = self.vit.heads(x)
        return x

if __name__ == '__main__':
    # model, weights = getViTModel()
    model = ViTBackbone(pretrained=True)
    vit, weights = model.vit, model.vit_weights
    image1 = read_image('dog.jpg')
    image2 = read_image('cat.jpg')
    image3 = read_image('car.jpg')
    batch = preprocess([image1, image2, image3], weights.transforms())
    # inference(model, batch, weights.meta["categories"])
    experiment(model, batch)
