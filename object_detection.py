import torch, torchvision
from torch import nn
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from time import time
import matplotlib.pyplot as plt

from utils import getVisualizableTransformedImageFromPIL, HWC2CHW
from models import preprocess

def getObjectDetectionModel():
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.8)
    model.eval()
    return model, weights

class ObjectDetection(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super().__init__()
        self.device = device
        self.detector, self.detector_weights = getObjectDetectionModel()  
        self.detector.to(self.device)

    def filterWithArea(self, prediction, area_threshold):
        boxes = prediction['boxes']
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights

        kept_ids = torch.nonzero(areas >= area_threshold, as_tuple=True)[0]
        prediction['boxes'] = prediction['boxes'][kept_ids]
        prediction['scores'] = prediction['scores'][kept_ids]
        prediction['labels'] = prediction['labels'][kept_ids] 
        return prediction            

    def inference(self, images, visualization=True):
        self.detector.eval()
        batch = preprocess(images, self.detector_weights.transforms())

        with torch.no_grad():
            batch = batch.to(self.device)
            predictions = self.detector.forward(batch)
            for i in range(len(predictions)):
                kept_ids = torchvision.ops.nms(predictions[i]['boxes'], predictions[i]['scores'], 0.01) 
                predictions[i]['boxes'] = predictions[i]['boxes'][kept_ids]
                predictions[i]['scores'] = predictions[i]['scores'][kept_ids]
                predictions[i]['labels'] = predictions[i]['labels'][kept_ids]
                predictions[i] = self.filterWithArea(predictions[i], 50 * 50)

        if visualization:
            for image, prediction in zip(images, predictions):
                labels = [self.detector_weights.meta['categories'][i] for i in prediction['labels']]
                box = draw_bounding_boxes(image, boxes=prediction['boxes'],
                                        labels=labels,
                                        colors='blue',
                                        width=1, font_size=30)
                plot = to_pil_image(box.detach())
                plt.figure()
                plt.imshow(plot)
                plt.show()

        return predictions

if __name__ == '__main__':
    image_paths = ['example_images/voc_example1.jpg', 'example_images/voc_example2.jpg', 'example_images/voc_example3.jpg']
    images = []
    vit_weights = ViT_B_16_Weights.DEFAULT
    for image_path in image_paths:
        image = Image.open(image_path)
        image = getVisualizableTransformedImageFromPIL(image, vit_weights.transforms())        
        image = HWC2CHW(image)
        images.append(image)

    object_detection = ObjectDetection()
    predictions = object_detection.inference(images, visualization=True)
    for p in predictions:
        print(p['boxes'])
        print('\n')
