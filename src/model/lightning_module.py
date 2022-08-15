import pytorch_lightning as pl
import torch.optim
import torch.nn as nn
import torch.nn.functional as F

class LitModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
