import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import os
import pdb
from time import time
from PIL import Image

from models import ViTBackbone
from utils import visualizePatchSimilarities, getVisualizableTransformedImageFromPIL, HWC2CHW
from sampling import labelsFromDetections, labelsFromMultiCrop, multiCropTransform
from object_detection import ObjectDetection
from losses import SupConLoss, cosineSimilarity

class ViTSSLTrainer():
    def __init__(self, task_type, train_dataloader, validate_dataloader, epoch_num, learning_rate, weight_decay, checkpoint_path='', test_dataloader=None, \
                 save_path='../trained_models', example_image_path='example_images/voc_example3.jpg', device=torch.device('cuda'), save_every=10):
        self.device = device
        self.task_type = task_type
        if self.task_type == 'o4p':
            self.model = ViTBackbone(device=self.device, pretrained=False)
        elif self.task_type == 'mp':
            self.model = ViTBackbone(device=self.device, pretrained=False)
        elif self.task_type == 'lp':
            self.model = ViTBackbone(device=self.device, pretrained=False)
        self.train_dataloader, self.validate_dataloader, self.test_dataloader = train_dataloader, validate_dataloader, test_dataloader
        self.epoch_num = epoch_num
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epoch_num * len(self.train_dataloader), eta_min=5e-5,
                                                                    last_epoch=-1)
        self.epoch = 0
        if checkpoint_path != '':
            self.load(checkpoint_path)

        self.save_path = save_path
        os.mkdir(self.save_path)
        self.save_every = save_every
        self.example_image_path = example_image_path
        self.writer = SummaryWriter(log_dir=self.save_path)
        if self.task_type == 'o4p' or self.task_type == 'mp':
            self.criterion = SupConLoss()
        elif self.task_type == 'lp':
            self.criterion = torch.nn.CrossEntropyLoss()
        self.object_detection = ObjectDetection(device=self.device)

    def snapshotStateDict(self, loss):
        state_dict = {
                        'epoch': self.epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'loss': loss,
                     }
        return state_dict

    def save(self, state_dict):
        save_path = os.path.join(self.save_path, str(self.epoch) + '.pt')
        torch.save(state_dict, save_path)
        print(f'saved state dict of {state_dict["epoch"]} to {save_path}')

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f'loaded state dict of {checkpoint["epoch"]} from {path}')

    def o4p(self, images):        
        with torch.no_grad():
            predictions = self.object_detection.inference(images.copy(), add_self_patch=True, visualization=False)
        features = self.model(images.copy(), feature_extraction=True)
        features = F.normalize(features, dim=-1)
        patch_size = int(self.model.vit.patch_size)
        features, labels = labelsFromDetections(features, predictions, patch_size)
        loss = self.criterion(features, labels=labels)
        return loss

    def multiCrop(self, images):
        batch_size = len(images)
        images = multiCropTransform(images, 2)
        features = self.model(images.copy(), feature_extraction=True, pooling=True)
        features = F.normalize(features, dim=-1)
        features, labels = labelsFromMultiCrop(features, batch_size)
        loss = self.criterion(features, labels=labels)
        if torch.isnan(loss).any():
            with torch.no_grad():
                predictions = object_detection.inference(images.copy(), visualization=True)
        assert not torch.isnan(loss).any(), 'loss is nan'        
        return loss

    def linearProbe(self, images, classes):
        logits = self.model(images, feature_extraction=False)
        loss = self.criterion(logits, classes)
        return loss

    def trainBatch(self, images, annotations):
        viz_transformed_images = []
        for image in images:
            image = getVisualizableTransformedImageFromPIL(image, self.model.vit_weights.transforms())
            image = HWC2CHW(image)
            viz_transformed_images.append(image)
        if self.task_type == 'o4p':
            loss = self.o4p(viz_transformed_images)
        elif self.task_type == 'mp':
            loss = self.multiCrop(viz_transformed_images)
        elif self.task_type == 'lp':
            labels = torch.tensor(annotations).to(self.device)
            loss = self.linearProbe(viz_transformed_images, labels)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()
        if self.epoch >= 1:
            self.scheduler.step()        
        return loss.item()

    def validateBatch(self, images, annotations):
        viz_transformed_images = []
        for image in images:
            image = getVisualizableTransformedImageFromPIL(image, self.model.vit_weights.transforms())
            image = HWC2CHW(image)
            viz_transformed_images.append(image)
        with torch.no_grad():
            if self.task_type == 'o4p':
                loss = self.o4p(viz_transformed_images)
            elif self.task_type == 'mp':
                loss = self.multiCrop(viz_transformed_images)
            elif self.task_type == 'lp':
                labels = torch.tensor(annotations).to(self.device)
                loss = self.linearProbe(viz_transformed_images, labels)                
        return loss.item()

    def visualizePatchSimilarities(self):
        self.model.eval()
        
        image = Image.open(self.example_image_path)
        images = [image]

        with torch.no_grad():
            features = self.model(images, feature_extraction=True)
            features = F.normalize(features, dim=-1)

            similarity_matrix = cosineSimilarity(features.squeeze(), softmax=True, temperature=1)
            similarity_matrix = similarity_matrix.detach().cpu()
            image = getVisualizableTransformedImageFromPIL(image, self.model.vit_weights.transforms())
            for patch_idx in range(75, 125, 3):
                plot_tensor = visualizePatchSimilarities(image, similarity_matrix, patch_idx, save=False)
                self.writer.add_image(f'patch_similarities_{patch_idx}', plot_tensor, self.epoch)
        
    def doEpoch(self, train):
        if train:
            self.model.train()
            dataloader = self.train_dataloader
        else:
            self.model.eval()
            dataloader = self.validate_dataloader
        epoch_loss = 0
        for batch_idx, batch_list in enumerate(dataloader):
            images = [x[0] for x in batch_list]
            annotations = [x[1] for x in batch_list]
            if train:
                loss_val = self.trainBatch(images, annotations)
            else:
                loss_val = self.validateBatch(images, annotations)
            epoch_loss += loss_val
            print(f'finished is train: {train} batch: {batch_idx} / {len(dataloader)}, loss: {loss_val}', end='\r', flush=True)
            # pdb.set_trace()
        epoch_loss /= len(self.train_dataloader)      
        return epoch_loss  

    def testLinearProbe(self):
        self.model.eval()  # Set the model to evaluation mode
        correct = 0

        assert self.test_dataloader.batch_size == 1, 'testing batch size should be 1'

        # Disable gradient computation to save memory and speed up the process
        correct = 0
        with torch.no_grad():
            for data in self.test_dataloader:
                image, label = data[0]

                image = getVisualizableTransformedImageFromPIL(image, self.model.vit_weights.transforms())
                image = HWC2CHW(image)

                label = torch.tensor([label]).to(self.device)

                outputs = self.model([image], feature_extraction=False)  # Forward pass
                outputs = outputs.squeeze()
                predicted = outputs.argmax()
                
                correct += (predicted == label.item()).sum().item()
                print('compare', correct, predicted, label.item())

        accuracy = correct / len(self.test_dataloader)
        return accuracy

    def train(self):
        min_validate_loss = np.inf
        best_state_dict = self.snapshotStateDict(-1)
        for epoch in range(self.epoch, self.epoch_num):
            print(f'\nepoch: {self.epoch} / {self.epoch_num}, loss: {min_validate_loss}')
            train_epoch_loss = self.doEpoch(True)
            validate_epoch_loss = self.doEpoch(False)

            if self.task_type == 'lp':
                accuracy = self.testLinearProbe()
                print(f'\nepoch: {self.epoch} / {self.epoch_num}, accuracy: {accuracy}')

            if validate_epoch_loss < min_validate_loss:
                min_validate_loss = validate_epoch_loss
                best_state_dict = self.snapshotStateDict(min_validate_loss)
            if self.epoch % self.save_every == 0:
                self.save(best_state_dict)

            self.writer.add_scalar('loss/train_epoch_loss', train_epoch_loss, global_step=self.epoch)
            self.writer.add_scalar('loss/validate_epoch_loss', validate_epoch_loss, global_step=self.epoch)
            if self.task_type == 'lp':
                self.writer.add_scalar('accuracy', accuracy, global_step=self.epoch)
            self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=self.epoch)
            if self.task_type == 'o4p' or self.task_type == 'mp':
                self.visualizePatchSimilarities()

            self.epoch += 1

if __name__ == '__main__':
    pass