import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate


class DNNModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.output(x)
        return x


class CNNModel(nn.Module):
    def __init__(self, nfeature):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=4)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 16, kernel_size=4)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * ((nfeature // 4) - 3), 32)  # Compute flattened input size manually
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # To have a shape of (batch_size, in_channels, sequence_length) ; sequence length is the len of time series
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.output(x)
        x = x.squeeze()
        return x


class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(16, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        ula, (h, _) = self.lstm(x)
        # Output and hidden state
        out = h[-1]  # Take the last output for prediction
        x = self.fc1(out)
        x = self.output(x)
        x = x.squeeze()
        return x


class SimplerModelModule(L.LightningModule):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.optimizer_cfg = self.cfg.models.two_step.model.optimizer
        self.model = model

    def training_step(self, batch, batch_idx):
        predicted_labels, target = self.forward(batch)
        loss = F.mse_loss(predicted_labels, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        predicted_labels, target = self.forward(batch)
        loss = F.mse_loss(predicted_labels, target)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer_cfg, params=self.parameters())
        return optimizer

    def predict_step(self, batch, batch_idx):
        predicted_labels, target = self.forward(batch)
        return predicted_labels

    def test_step(self, batch, batch_idx):
        predicted_labels, target = self.forward(batch)
        return predicted_labels

    def forward(self, batch):
        peaks, target = batch
        predicted_labels = self.model(peaks).squeeze()
        return predicted_labels, target


class RNNModule(SimplerModelModule):
    def __init__(cfg):
        model = RNNModel()
        super().__init__(cfg, model=model)


class DNNModule(SimplerModelModule):
    def __init__(cfg):
        model = DNNModel(cfg=cfg)
        super().__init__(lr=cfg, model=model)


class CNNModule(SimplerModelModule):
    def __init__(cfg):
        model = CNNModel(cfg=cfg)
        super().__init__(lr=cfg, model=model)