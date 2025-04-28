import torch
import lightning as L
import torch.optim as optim
import torch.nn.functional as F



class LSTM(torch.nn.Module):  # TODO: Is this implemented like in their paper? In their paper they have multiple LSTMs.
    def __init__(self, input_dim: int = 3000, lstm_hidden_dim: int = 32, num_lstm_layers: int = 1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1, num_layers=num_lstm_layers, hidden_size=lstm_hidden_dim, batch_first=True)
        self.fc3 = torch.nn.Linear(lstm_hidden_dim, 32)
        self.fc4 = torch.nn.Linear(32, 1)

    def forward(self, x):
        ula, (h, _) = self.lstm(x)
        out = h[-1]
        out = F.relu(self.fc3(out))  # If we would like to have a prediction for each point in wf, then we would use ula instead of out here
        clf = F.sigmoid(self.fc4(out)).squeeze()
        return clf


class LSTMModule(L.LightningModule):
    def __init__(self, name: str, hyperparameters: dict):
        self.name = name
        self.hyperparameters = hyperparameters
        super().__init__()
        self.lstm = LSTM(
            input_dim=self.hyperparameters["input_dim"],
            lstm_hidden_dim=self.hyperparameters["lstm_hidden_dim"],
            num_lstm_layers=self.hyperparameters["num_lstm_layers"]
        )

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
        return optim.AdamW(self.parameters(), lr=0.001)

    def predict_step(self, batch, batch_idx):
        predicted_labels, target = self.forward(batch)
        return predicted_labels

    def test_step(self, batch, batch_idx):
        predicted_labels, target = self.forward(batch)
        return predicted_labels

    def forward(self, batch):
        waveform, target, wf_idx = batch
        predicted_labels = self.lstm(waveform).squeeze()
        return predicted_labels, target

    # def validation_epoch_end(self, val_step_outputs):
    #     for pred in val_step_outputs:
    #         print(pred)
    # do something with all the predictions from each validation_step

    # def configure_optimizer(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    #     return optimizer
