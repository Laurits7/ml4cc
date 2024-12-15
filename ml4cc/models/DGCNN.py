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

    def forward(self, data):
        x, batch = data.x, data.batch
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
        waveform, target = batch
        predicted = self.lstm(waveform)
        loss = F.nll_loss(predicted, target)
        return loss

    def validation_step(self, batch, batch_idx):
        waveform, target = batch
        predicted = self.lstm(waveform)
        loss = F.nll_loss(predicted, target)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


# def train():
#     model.train()

#     total_loss = 0
#     correct = 0
#     total = 0
#     for data in train_loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         out = model(data)
#         loss = F.nll_loss(out, data.y.long())
#         loss.backward()
#         total_loss += loss.item() * data.num_graphs

#         correct += out.max(dim=1)[1].eq(data.y).sum().item()
#         total += data.num_nodes

#         optimizer.step()
#     return total_loss / train_dataset.__len__(), correct/total
