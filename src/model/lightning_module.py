import pytorch_lightning as pl
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
from torch import nn

class LitModel(pl):
    def __init__(self, model):
        super().__init__()
        self.classifier = nn.Linear(768, self.num_classes)
        self.model = nn.Sequential(self.feature_extractor, self.classifier)
        self.feature_extractor = timm.create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes=0)

        config = resolve_data_config({}, model=self.feature_extractor)
        transform = create_transform(**config)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)