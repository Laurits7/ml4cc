import torch
import lightning as L
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# TODO: Is this implemented like in their paper? In their paper they have
# multiple LSTMs.
class LSTM(torch.nn.Module):
    def __init__(self, num_features, lstm_hidden_dim: int = 32, num_lstm_layers: int = 1):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=num_features, num_layers=num_lstm_layers, hidden_size=lstm_hidden_dim, batch_first=True
        )
        self.fc3 = torch.nn.Linear(lstm_hidden_dim, 32)
        self.fc4 = torch.nn.Linear(32, 1)

    def forward(self, x, mask):
        lengths = mask.sum(dim=1)
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        ula, (h, _) = self.lstm(packed_x)
        # Unpack sequence to apply linear layer to each timestep
        padded_out, _ = pad_packed_sequence(ula, batch_first=True)
        out = F.relu(self.fc3(padded_out))
        out = self.fc4(out)
        out = torch.sigmoid(out)  # use torch.sigmoid, not F.sigmoid (deprecated)
        return out


class LSTMModule(L.LightningModule):
    def __init__(self, name: str, hyperparameters: dict, checkpoint: dict = None):
        self.name = name
        self.hyperparameters = hyperparameters
        super().__init__()
        self.lstm = LSTM(
            num_features=self.hyperparameters["num_features"],
            lstm_hidden_dim=self.hyperparameters["lstm_hidden_dim"],
            num_lstm_layers=self.hyperparameters["num_lstm_layers"]
        )

    def training_step(self, batch, batch_idx):
        predicted_labels, target = self.forward(batch)
        target_mask = (target != -1) & (target != -999)
        masked_target = target[target_mask]
        masked_predicted_labels = predicted_labels[target_mask]
        loss = F.binary_cross_entropy(masked_predicted_labels, masked_target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        predicted_labels, target = self.forward(batch)
        target_mask = (target != -1) & (target != -999)
        masked_target = target[target_mask]
        masked_predicted_labels = predicted_labels[target_mask]
        loss = F.binary_cross_entropy(masked_predicted_labels, masked_target)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.001)
        scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def predict_step(self, batch, batch_idx):
        predicted_labels, _ = self.forward(batch)
        return predicted_labels

    def test_step(self, batch, batch_idx):
        predicted_labels, _ = self.forward(batch)
        return predicted_labels

    def forward(self, batch):
        waveform, target, mask = batch
        predicted_labels = self.lstm(waveform, mask).squeeze(dim=-1)
        return predicted_labels, target
