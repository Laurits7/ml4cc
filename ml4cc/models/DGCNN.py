import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L


class DGCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = DynamicEdgeConv(MLP([2 * 1      , n_conv1, n_conv1]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * n_conv1, n_conv2, n_conv2]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * n_conv2, n_conv3, n_conv3]), k, aggr)
        self.mlp = MLP([n_conv1+n_conv2+n_conv3, n_mlp1, n_mlp2, n_mlp3, out_channels], dropout=0.5, norm=None)

    def forward(self, data):
        x, batch = data.x, data.batch
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        out = self.mlp(torch.cat([x1, x2, x3], dim=1))

        return F.log_softmax(out, dim=1)
