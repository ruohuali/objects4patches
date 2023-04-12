import torch, torchvision
from torchvision.io import read_image
from torchvision.models import vit_b_16, ViT_B_16_Weights

def getViTModel(pretrained=False):
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(None)
    # model = vit_b_16()
    return model, weights

def preprocess(images, model_preprocess):
    batch = []
    for image in images:
        x = model_preprocess(image)
        batch.append(x)
    batch = torch.stack(batch)    
    return batch

def inference(model, batch, class_names, topk=3):
    model.eval()
    logits = model(batch)
    predictions = logits.softmax(dim=1)
    for prediction in predictions:
        topk_ids = torch.topk(prediction, topk).indices
        for idx in topk_ids:
            score = prediction[idx].item()
            category_name = class_names[idx]
            print(f"{category_name}: {100 * score:.1f}%")
        print('-' * 50)

def extract_feature(model, batch):
    model.eval()
    logits = model(batch)

if __name__ == '__main__':
    model, weights = getViTModel()
    image1 = read_image('dog.jpg')
    image2 = read_image('cat.jpg')
    image3 = read_image('car.jpg')
    batch = preprocess([image1, image2, image3], weights.transforms())
    inference(model, batch, weights.meta["categories"])