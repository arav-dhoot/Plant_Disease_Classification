import pytorch_lightning as pl
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
import numpy as np

class LitModel(LightningModule):
    def __init__(self, num_classes, model):
        super().__init__()
        self.num_classes = num_classes
        self.model = model
        self.classifier = nn.Linear(768, self.num_classes)
        if model == 'ViT':
            self.feature_extractor = timm.create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes=0)
            config = resolve_data_config({}, model=self.feature_extractor)
            transform = create_transform(**config)
        elif model == 'ResNet':
            pass
            # Create a ResNet model
        self.criterion =  nn.CrossEntropyLoss()
        self.model = nn.Sequential(self.feature_extractor, self.classifier)

    def training_step(self, batch, batch_idx):
        x, y = batch    
        representations = self.feature_extractor(x)
        logits = self.classifier(representations)
        probabilities = torch.softmax(logits, dim=1)
        y = torch.tensor(y)
        loss = self.criterion(probabilities, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        representations = self.feature_extractor(x)
        logits = self.classifier(representations)
        probabilities = torch.softmax(logits, dim=1)
        print(y)
        print(type(y))
        y = np.array(y)
        y = torch.from_numpy(y).long()
        loss = self.criterion(probabilities, y)
        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # def forward(self, x):
        # return torch.relu(self.l1(x.view(x.size(0), -1)))
                         
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)