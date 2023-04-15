import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import ViTBackbone
from utils import visualizePatchSimilarities
from sampling import labelsFromDetections
from object_detection import ObjectDetection

class SSLTrainer():
    def __init__(self, model_type, train_loader, validate_loader, epoch_num, learning_rate, weight_decay, checkpoint_path=None, \
                 device=torch.device('cuda')):
        self.device = device
        self.model_type = model_type
        if self.model_type == 'ViTBackbone':
            self.model = ViTBackbone(pretrained=False).to(self.device)
        self.train_loader, self.validate_loader = train_loader, validate_loader
        self.epoch_num = epoch_num
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train_loader) * self.epoch_num, eta_min=0,
                                                                    last_epoch=-1)
        if checkpoint_path is not None:
            self.load(checkpoint_path)

        self.writer = SummaryWriter()

    def save(self, loss):
        torch.save({
                    'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': loss,
                    }, self.save_path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    def o4p(self, images):
        images

    def trainEpoch(self, images):
        pass

    def visualizePatchSimilarities(self):
        pass

    def train(self):
        self.model.train()
        for epoch in range(self.epoch, self.epoch_num):
            epoch_loss = 0
            for batch_idx, batch in enumerate(self.dataloader):
                loss_val = self.trainEpoch(batch)
                epoch_loss += loss_val

         