import torch
import lightning as L
import torch.optim as optim
import torch.nn.functional as F


class LSTM(torch.nn.Module):
    def __init__(self, lstm_hidden_dim: int = 32):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1, num_layers=1, hidden_size=lstm_hidden_dim, batch_first=True)
        self.fc3 = torch.nn.Linear(lstm_hidden_dim, 32)
        self.fc4 = torch.nn.Linear(32, 1)

    def forward(self, x):
        ula, _ = self.lstm(x)
        out = F.relu(self.fc3(ula))
        clf = F.sigmoid(self.fc4(out))
        return clf


class LSTMModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.lstm = LSTM()

    def training_step(self, batch, batch_idx):
        waveform, target = batch
        predicted_labels = self.lstm(waveform)
        loss = F.mse_loss(predicted_labels, target)
        return loss

    def validation_step(self, batch, batch_idx):
        waveform, target = batch
        predicted_labels = self.lstm(waveform)
        loss = F.mse_loss(predicted_labels, target)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

    # def validation_epoch_end(self, val_step_outputs):
    #     for pred in val_step_outputs:
    #         print(pred)
    # do something with all the predictions from each validation_step

    # def configure_optimizer(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    #     return optimizer
