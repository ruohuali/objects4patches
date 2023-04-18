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
        self.detector.eval()

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

    def filterWithNMS(self, prediction, iou_threshold=0.01):
        kept_ids = torchvision.ops.nms(prediction['boxes'], prediction['scores'], iou_threshold) 
        prediction['boxes'] = prediction['boxes'][kept_ids]
        prediction['scores'] = prediction['scores'][kept_ids]
        prediction['labels'] = prediction['labels'][kept_ids]
        return prediction

    def addSelfPatch2Prediction(self, prediction, height, width, patch_size, num_patches):
        # Generate random starting coordinates for patches
        rand_top_left_x = torch.randint(0, width - patch_size, (num_patches,))
        rand_top_left_y = torch.randint(0, height - patch_size, (num_patches,))

        patch_indices = []

        for i in range(num_patches):
            x_start, y_start = rand_top_left_x[i], rand_top_left_y[i]
            x_end, y_end = x_start + patch_size, y_start + patch_size
            patch_indices.append([x_start, y_start, x_end, y_end])

        # Convert patch_indices list to a tensor
        patch_indices_tensor = torch.tensor(patch_indices).to(self.device)
        patch_scores_tensor = torch.tensor([0.1 for _ in range(len(patch_indices_tensor))]).to(self.device)
        patch_labels_tensor = torch.tensor([0 for _ in range(len(patch_indices_tensor))]).to(self.device)

        prediction['boxes'] = torch.cat((prediction['boxes'], patch_indices_tensor), dim=0)   
        prediction['scores'] = torch.cat((prediction['scores'], patch_scores_tensor), dim=0)   
        prediction['labels'] = torch.cat((prediction['labels'], patch_labels_tensor), dim=0)   
        return prediction

    def inference(self, images, add_self_patch=False, visualization=True):
        batch = preprocess(images, self.detector_weights.transforms())

        with torch.no_grad():
            batch = batch.to(self.device)
            predictions = self.detector.forward(batch)
            for i in range(len(predictions)):
                if add_self_patch:
                    predictions[i] = self.addSelfPatch2Prediction(predictions[i], batch.size(-2), batch.size(-1), 50, 30)
                predictions[i] = self.filterWithNMS(predictions[i])
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
    predictions = object_detection.inference(images, add_self_patch=True, visualization=True)
    for p in predictions:
        print(p['boxes'])
        print('\n')
