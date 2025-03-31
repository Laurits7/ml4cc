import torch
from torch import nn
import lightning as L
import torch.optim as optim
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.nn import MLP, DynamicEdgeConv


class DGCNN(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.conv1 = DynamicEdgeConv(MLP([2 * 1, cfg.n_conv1, cfg.n_conv1]), cfg.k, cfg.aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * cfg.n_conv1, cfg.n_conv2, cfg.n_conv2]), cfg.k, cfg.aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * cfg.n_conv2, cfg.n_conv3, cfg.n_conv3]), cfg.k, cfg.aggr)
        self.mlp = MLP(
            [cfg.n_conv1 + cfg.n_conv2 + cfg.n_conv3, cfg.n_mlp1, cfg.n_mlp2, cfg.n_mlp3, cfg.out_channels],
            dropout=cfg.mlp_dropout,
            norm=None,
        )

    def forward(self, x, batch):
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        out = self.mlp(torch.cat([x1, x2, x3], dim=1))
        return F.log_softmax(out, dim=1)


class DGCNNModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dgcnn = DGCNN(cfg=cfg)

    def training_step(self, batch, batch_idx):
        predicted_labels, target = self.forward(batch, batch_idx)
        loss = F.nll_loss(predicted_labels, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        predicted_labels, target = self.forward(batch, batch_idx)
        loss = F.nll_loss(predicted_labels, target)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=0.001)

    def forward(self, batch, batch_idx):
        peaks, target = batch
        batch_indices = [batch_idx] * len(target)
        predicted_labels = self.dgcnn(peaks, batch_indices).squeeze()
        return predicted_labels, target
