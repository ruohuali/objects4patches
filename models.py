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

class ViTBackbone(nn.Module):
    def __init__(self, device=torch.device('cuda'), pretrained=False):
        super().__init__()
        self.vit, self.vit_weights = getViTModel(pretrained=pretrained)  
        self.device = device 
        self.vit.to(self.device)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, images, feature_extraction=True, pooling=False):
        x = preprocess(images, self.vit_weights.transforms())
        x = x.to(self.device)

        # Reshape and permute the input tensor
        x = self.vit._process_input(x)
        n = x.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.vit.encoder(x)

        if feature_extraction:
            if pooling:
                y = self.avg_pool(x[:, 1:].permute(0, 2, 1)).permute(0, 2, 1)
            else:
                y = x[:, 1:]
            return y

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        x = self.vit.heads(x)
        return x

    def inference(self, images, topk=3, visualization=True):
        self.vit.eval()
        with torch.no_grad():
            logits = self.forward(images, feature_extraction=False)
        predictions = logits.softmax(dim=1)
        if visualization:
            for prediction in predictions:
                topk_ids = torch.topk(prediction, topk).indices
                for idx in topk_ids:
                    score = prediction[idx].item()
                    category_name = self.vit_weights.meta["categories"][idx]
                    print(f"{category_name}: {100 * score:.1f}%")
                print('-' * 50) 
        return predictions

if __name__ == '__main__':
    pass


