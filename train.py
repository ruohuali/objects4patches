import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import os
import pdb

from models import ViTBackbone
from utils import visualizePatchSimilarities, getVisualizableTransformedImageFromPIL, HWC2CHW
from sampling import labelsFromDetections
from object_detection import ObjectDetection
from losses import SupConLoss, cosineSimilarity

class ViTSSLTrainer():
    def __init__(self, task_type, train_loader, validate_loader, epoch_num, learning_rate, weight_decay, checkpoint_path='', \
                 save_path='../trained_models', example_image_path='example_images/voc_example3.jpg', device=torch.device('cuda'), save_every=10):
        self.device = device
        self.task_type = task_type
        if self.task_type == 'o4p':
            self.model = ViTBackbone(device=self.device, pretrained=False)
        self.train_loader, self.validate_loader = train_loader, validate_loader
        self.epoch_num = epoch_num
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epoch_num * len(self.train_loader) * 2, eta_min=5e-5,
                                                                    last_epoch=-1)
        self.epoch = 0
        if checkpoint_path != '':
            self.load(checkpoint_path)

        self.save_path = save_path
        os.mkdir(self.save_path)
        self.save_every = save_every
        self.writer = SummaryWriter(log_dir=self.save_path)
        self.criterion = SupConLoss()

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

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    def o4p(self, images):        
        object_detection = ObjectDetection()
        with torch.no_grad():
            predictions = object_detection.inference(images.copy(), visualization=True)

        features = self.model(images.copy(), feature_extraction=True)
        assert not torch.isnan(features).any(), 'features'
        features = F.normalize(features, dim=-1)
        assert not torch.isnan(features).any(), 'features norm'

        patch_size = int(self.model.vit.patch_size)
        features, labels = labelsFromDetections(features, predictions, patch_size)
        assert not torch.isnan(features).any(), 'features reshape'
        assert not torch.isnan(labels).any(), 'labels'
        loss = self.criterion(features, labels=labels)
        assert not torch.isnan(loss).any(), 'loss'
        return loss

    def trainBatch(self, images):
        viz_transformed_images = []
        for image in images:
            image = getVisualizableTransformedImageFromPIL(image, self.model.vit_weights.transforms())
            image = HWC2CHW(image)
            viz_transformed_images.append(image)
        # with torch.autograd.detect_anomaly():            
        if self.task_type == 'o4p':
            loss = self.o4p(viz_transformed_images)
        self.optimizer.zero_grad()
        loss.backward()

        grad_sum = 0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_sum += torch.sum(param.grad).item()
        print('grad_sum', grad_sum)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()

        if self.epoch >= 10:
            self.scheduler.step()        
        return loss.item()

    def validateBatch(self, images):
        viz_transformed_images = []
        for image in images:
            image = getVisualizableTransformedImageFromPIL(image, model.vit_weights.transforms())
            image = HWC2CHW(image)
            viz_transformed_images.append(image)
        with torch.no_grad():
            loss = self.o4p(viz_transformed_images)
        return loss.item()

    def visualizePatchSimilarities(self):
        self.model.eval()
        
        image = Image.open(self.example_image_path)
        images = [image]

        with torch.no_grad():
            features = model(images, feature_extraction=True)
            features = F.normalize(features, dim=-1)

            similarity_matrix = cosineSimilarity(features.squeeze(), softmax=True, temperature=1)
            image = getVisualizableTransformedImageFromPIL(image, self.model.vit_weights.transforms())
            for patch_idx in range(75, 125, 3):
                plot_tensor = visualizePatchSimilarities(image, similarity_matrix, patch_idx, save=False)
                self.writer.add_image(f'patch_similarities_{patch_idx}', plot_tensor, self.epoch)
        
    def doEpoch(self, train):
        if train:
            self.model.train()
            dataloader = self.train_loader
        else:
            self.model.eval()
            dataloader = self.validate_loader
        epoch_loss = 0
        for batch_idx, batch_list in enumerate(dataloader):
            print('start', batch_idx)
            images = [x[0] for x in batch_list]
            if train:
                loss_val = self.trainBatch(images)
            else:
                loss_val = self.validateBatch(images)
            epoch_loss += loss_val
            print(f'is train: {train} batch: {batch_idx} / {len(dataloader)}, loss: {loss_val}', end='\n')
            pdb.set_trace()
        epoch_loss /= len(self.train_loader)      
        return epoch_loss  

    def train(self):
        min_validate_loss = np.inf
        best_state_dict = self.snapshotStateDict(-1)
        for epoch in range(self.epoch, self.epoch_num):
            print(f'\nepoch: {self.epoch} / {self.epoch_num}, loss: {min_validate_loss}')
            train_epoch_loss = self.doEpoch(True)
            validate_epoch_loss = self.doEpoch(False)

            if validate_epoch_loss < min_validate_loss:
                min_validate_loss = validate_epoch_loss
                best_state_dict = self.snapshotStateDict(min_validate_loss)
            if self.epoch % self.save_every == 0:
                self.save(best_state_dict)
                print(f'saved model at {self.save_path}')

            self.writer.add_scalar('train_epoch_loss', train_epoch_loss, global_step=self.epoch)
            self.writer.add_scalar('validate_epoch_loss', validate_epoch_loss, global_step=self.epoch)
            self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=self.epoch)
            self.visualizePatchSimilarities()

            self.epoch += 1

if __name__ == '__main__':
    pass