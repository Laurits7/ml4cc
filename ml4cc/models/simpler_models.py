import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pack_padded_sequence
# from hydra.utils import instantiate
import importlib
from omegaconf import OmegaConf
import torch.optim as optim


def resolve_target(target_str):
    """Resolve a string like class to the actual class."""
    module_path, class_name = target_str.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class DNNModel(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.fc1 = nn.Linear(
            in_features=hyperparameters.n_features * 2,  # As the mask is concatenated
            out_features=hyperparameters.linear_layer_1.out_features
        )
        self.fc2 = nn.Linear(
            in_features=hyperparameters.linear_layer_1.out_features,
            out_features=hyperparameters.linear_layer_2.out_features,
        )
        self.fc3 = nn.Linear(
            in_features=hyperparameters.linear_layer_2.out_features,
            out_features=hyperparameters.linear_layer_3.out_features,
        )
        self.output = nn.Linear(
            in_features=hyperparameters.linear_layer_3.out_features,
            out_features=hyperparameters.output_layer.out_features,
        )

    def forward(self, x, mask):
        mask = mask.float()
        x = torch.concat([x, mask], axis=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.output(x)
        return x


class CNNModel(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=hyperparameters.conv_layer_1.in_channels,
            out_channels=hyperparameters.conv_layer_1.out_channels,
            kernel_size=hyperparameters.conv_layer_1.kernel_size,
        )
        self.pool1 = nn.MaxPool1d(kernel_size=hyperparameters.pool_layer_1.kernel_size)
        self.conv2 = nn.Conv1d(
            in_channels=hyperparameters.conv_layer_1.out_channels,
            out_channels=hyperparameters.conv_layer_2.out_channels,
            kernel_size=hyperparameters.conv_layer_2.kernel_size,
        )
        self.pool2 = nn.MaxPool1d(kernel_size=hyperparameters.pool_layer_2.kernel_size)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(
            # Compute flattened input size manually
            in_features=6560,
            out_features=hyperparameters.linear_layer_1.out_features,
        )
        self.output = nn.Linear(
            in_features=hyperparameters.linear_layer_1.out_features,
            out_features=hyperparameters.output_layer.out_features,
        )

    def forward(self, x, mask):
        # To have a shape of (batch_size, in_channels, sequence_length) ;
        # sequence length is the len of time series
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        if mask is not None:
            # Resize the mask to match x.shape[-1]
            # Use interpolation to safely downsample the mask
            mask = mask.unsqueeze(1).float()  # (B, 1, 1650)
            mask = F.interpolate(mask, size=x.shape[-1], mode='nearest')  # (B, 1, new_len)

            # Apply the mask
            x = x * mask

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.output(x)
        x = x.squeeze()
        return x


class RNNModel(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hyperparameters.LSTM_layers.input_size,
            hidden_size=hyperparameters.LSTM_layers.hidden_size,
            num_layers=hyperparameters.LSTM_layers.num_layers,
            batch_first=hyperparameters.LSTM_layers.batch_first,
        )
        self.fc1 = nn.Linear(
            in_features=hyperparameters.LSTM_layers.hidden_size,
            out_features=hyperparameters.linear_layer_1.out_features,
        )
        self.output = nn.Linear(
            in_features=hyperparameters.linear_layer_1.out_features,
            out_features=hyperparameters.output_layer.out_features,
        )

    def forward(self, x, mask):
        x = x.unsqueeze(-1)
        lengths = mask.sum(dim=1)
        # Pack the padded sequence
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True,enforce_sorted=False)
        ula, (h, _) = self.lstm(x)
        # Output and hidden state
        out = h[-1]  # Take the last output for prediction
        x = self.fc1(out)
        x = self.output(x)
        x = x.squeeze()
        return x


class SimplerModelModule(L.LightningModule):
    def __init__(self, optimizer_cfg, model):
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
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
        optimizer = optim.AdamW(self.parameters(), lr=0.001)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def predict_step(self, batch, batch_idx):
        predicted_labels, _ = self.forward(batch)
        return predicted_labels

    def test_step(self, batch, batch_idx):
        predicted_labels, _ = self.forward(batch)
        return predicted_labels

    def forward(self, batch):
        peaks, target, mask = batch  # In principle, we should include a mask to assist in the training
        predicted_labels = self.model(peaks, mask).squeeze(-1)
        return predicted_labels, target


class RNNModule(SimplerModelModule):
    def __init__(self, name: str, hyperparameters: dict, optimizer: dict, checkpoint: dict):
        self.name = name
        self.checkpoint = checkpoint
        self.optimizer_cfg = optimizer
        self.hyperparameters = OmegaConf.create(hyperparameters)
        model = RNNModel(hyperparameters=hyperparameters)
        super().__init__(optimizer_cfg=self.optimizer_cfg, model=model)


class DNNModule(SimplerModelModule):
    def __init__(self, name: str, hyperparameters: dict, optimizer: dict, checkpoint: dict):
        self.name = name
        self.checkpoint = checkpoint
        self.optimizer_cfg = optimizer
        self.hyperparameters = OmegaConf.create(hyperparameters)
        model = DNNModel(hyperparameters=hyperparameters)
        super().__init__(optimizer_cfg=self.optimizer_cfg, model=model)


class CNNModule(SimplerModelModule):
    def __init__(self, name: str, hyperparameters: dict, optimizer: dict, checkpoint: dict):
        self.name = name
        self.checkpoint = checkpoint
        self.optimizer_cfg = optimizer
        self.hyperparameters = OmegaConf.create(hyperparameters)
        model = CNNModel(hyperparameters=self.hyperparameters)
        super().__init__(optimizer_cfg=self.optimizer_cfg, model=model)
