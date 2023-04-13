from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

from utils import getVisualizableTransformedImage

image_path = 'coco_example.jpg'

img = read_image(image_path)
print(img.shape, img.dtype)

weights1 = ViT_B_16_Weights.DEFAULT
img1 = getVisualizableTransformedImage(image_path, weights1.transforms())
img1 = img1.permute(2, 0, 1)
print(img1.shape)

# Step 1: Initialize model with the best available weights
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.8)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = [preprocess(img1)]

# Step 4: Use the model and visualize the prediction
prediction = model(batch)[0]
labels = [weights.meta["categories"][i] for i in prediction["labels"]]
box = draw_bounding_boxes(img1, boxes=prediction["boxes"],
                          labels=labels,
                          colors="red",
                          width=1, font_size=30)
im = to_pil_image(box.detach())
im.show()