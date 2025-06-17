import torch
import lightning as L
import torch.optim as optim
import torch.nn.functional as F


# TODO: Is this implemented like in their paper? In their paper they have
# multiple LSTMs.
class LSTM(torch.nn.Module):
    def __init__(self, num_features, lstm_hidden_dim: int = 32, num_lstm_layers: int = 1, regression: bool = False):
        super().__init__()
        self.regression = regression
        self.lstm = torch.nn.LSTM(
            input_size=num_features, num_layers=num_lstm_layers,hidden_size=lstm_hidden_dim, batch_first=True
        )
        self.fc3 = torch.nn.Linear(lstm_hidden_dim, 32)
        self.fc4 = torch.nn.Linear(32, 1)

    def forward(self, x):
        ula, (h, _) = self.lstm(x)
        out = h[-1] if self.regression else ula
        # If we would like to have a prediction for each point in wf, then we
        # would use ula instead of out here
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        clf = out if self.regression else F.sigmoid(out)
        return clf


class LSTMModule(L.LightningModule):
    def __init__(self, name: str, hyperparameters: dict):
        self.name = name
        self.hyperparameters = hyperparameters
        super().__init__()
        self.lstm = LSTM(
            num_features=self.hyperparameters["num_features"],
            lstm_hidden_dim=self.hyperparameters["lstm_hidden_dim"],
            num_lstm_layers=self.hyperparameters["num_lstm_layers"],
            regression=self.hyperparameters.get("regression", False),
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
        masked_target = target[target != -1]
        masked_predicted_labels = predicted_labels[target != -1]
        loss = F.binary_cross_entropy(masked_predicted_labels, masked_target)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=0.001)

    def predict_step(self, batch, batch_idx):
        predicted_labels, _ = self.forward(batch)
        return predicted_labels

    def test_step(self, batch, batch_idx):
        predicted_labels, _ = self.forward(batch)
        return predicted_labels

    def forward(self, batch):
        waveform, target = batch
        predicted_labels = self.lstm(waveform).squeeze(dim=-1)
        return predicted_labels, target
