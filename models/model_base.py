import numpy as np

from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torch

# from benchmarks.validation_metrics import LeadTimeEval
# from benchmarks.unet import UNet


class ModelBase(pl.LightningModule):
    
    def __init__(
        self, 
        lr,
        weight_decay,
    ):
        
        super(ModelBase, self).__init__()
        
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay

        self.main_metric = 'mse'
        self.loss_fn = {'smoothL1': nn.SmoothL1Loss(), 
                        'L1': nn.L1Loss(), 
                        'mse': F.mse_loss}
        
        self.loss_fn = self.loss_fn[self.main_metric]
            
    def forward(self, x):
        pass
    
    def _compute_loss(self, y_hat, y, agg=True):
        if agg:
            loss = self.loss_fn(y_hat, y)
        else:
            loss = self.loss_fn(y_hat, y, reduction='none')
        return loss
    
    def training_step(self, batch, batch_idx):
        pass
        
    def validation_step(self, batch, batch_idx):
        pass
          
    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }        
