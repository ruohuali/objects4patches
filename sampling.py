import torch
from PIL import Image
import torch.optim as optim

from object_detection import ObjectDetection
from models import ViTBackbone
from utils import getVisualizableTransformedImageFromPIL, visualizeLabels, HWC2CHW
from losses import SupConLoss

def pixelIdx2PatchIdx(pixel_idx, patch_size):
    '''
    @param pixel_idx ~ (N, 2)
    '''
    x, y = pixel_idx
    j = int(x.item() // patch_size)
    i = int(y.item() // patch_size)
    return i, j

def makeLabelFromDetection(prediction, row_num, patch_size, running_class_id):
    label = torch.ones(row_num, row_num, dtype=torch.long) * running_class_id
    running_class_id += 1
    for box in prediction['boxes']:
        lt_pixel_idx, rb_pixel_idx = box[:2], box[2:]
        lt_i, lt_j = pixelIdx2PatchIdx(lt_pixel_idx, patch_size)
        rb_i, rb_j = pixelIdx2PatchIdx(rb_pixel_idx, patch_size)
        label[lt_i:rb_i, lt_j:rb_j] = running_class_id
        running_class_id += 1
    label = label.view(row_num * row_num)
    return label, running_class_id

def labelsFromDetections(features, predictions, patch_size):
    '''
    @param features ~ (B, P^2, D)
    @param predictions ~ [
        {
            'boxes' ~ (K, 4)
            'labels'
            'scores'
        }
        ...
    ]
    @return features ~ (B*P^2, 1, D)
    @return labels ~ (B*P^2)
    '''
    row_num = int(features.size(1)**(1 / 2))
    labels = []
    running_class_id = 0
    for prediction in predictions:
        label, running_class_id = makeLabelFromDetection(prediction, row_num, patch_size, running_class_id)
        labels.append(label)
    labels = torch.cat(labels)
    features = features.contiguous().view(features.size(0) * features.size(1), features.size(2))
    features = features.unsqueeze(1)
    return features, labels
            
if __name__ == '__main__':
    image_paths = ['example_images/coco_example1.jpg', 'example_images/coco_example2.jpg', 'example_images/voc_example1.jpg']
    images = []

    model = ViTBackbone(pretrained=True)
    
    for image_path in image_paths:
        image = Image.open(image_path)
        image = getVisualizableTransformedImageFromPIL(image, model.vit_weights.transforms())
        image = HWC2CHW(image)
        images.append(image)

    object_detection = ObjectDetection()
    predictions = object_detection.inference(images, visualization=False)

    features = model(images, feature_extraction=True)
    B, P2, D = features.size()

    patch_size = int(model.vit.patch_size)
    print('ps', patch_size)
    features, labels = labelsFromDetections(features, predictions, patch_size)
    print(features.shape, labels.shape)

    visualizeLabels(images, labels, (B, P2, D))

    criterion = SupConLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer.zero_grad()

    loss = criterion(features, labels=labels)

    loss.backward()
    optimizer.step()