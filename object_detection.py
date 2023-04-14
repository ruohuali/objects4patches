import torch
from torch import nn
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import time

from utils import getVisualizableTransformedImage
from models import preprocess

def getObjectDetectionModel():
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.8)
    model.eval()
    return model, weights

class ObjectDetection(nn.Module):
    def __init__(self):
        super().__init__()
        self.detector, self.detector_weights = getObjectDetectionModel()  

    def inference(self, images, visualization=True):
        self.detector.eval()
        batch = preprocess(images, self.detector_weights.transforms())
        with torch.no_grad():
            predictions = self.detector.forward(batch)
        if visualization:
            for image, prediction in zip(images, predictions):
                labels = [self.detector_weights.meta['categories'][i] for i in prediction['labels']]
                box = draw_bounding_boxes(image, boxes=prediction['boxes'],
                                        labels=labels,
                                        colors='blue',
                                        width=1, font_size=30)
                plot = to_pil_image(box.detach())
                plot.show()
        return predictions

if __name__ == '__main__':
    image_paths = ['example_images/voc_example1.jpg', 'example_images/voc_example2.jpg', 'example_images/voc_example3.jpg']
    images = []
    vit_weights = ViT_B_16_Weights.DEFAULT
    for image_path in image_paths:
        image = getVisualizableTransformedImage(image_path, vit_weights.transforms())
        image = image.permute(2, 0, 1)
        images.append(image)

    object_detection = ObjectDetection()
    predictions = object_detection.inference(images, visualization=True)
    for p in predictions:
        print(p['boxes'])
        print('\n')
