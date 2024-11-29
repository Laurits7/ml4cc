import os
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1, num_layers=1, hidden_size=32, batch_first=True)
        self.fc3 = torch.nn.Linear(32, 32)
        self.fc4 = torch.nn.Linear(32, 1)

    def forward(self, x):
        ula, (h, _) = self.lstm(x)
        out = h[-1]
        out = F.relu(self.fc3(out))
        clf = F.sigmoid(self.fc4(out))
        return clf


class LSTMModule(L.LightningModule):
    def __init__(self):
        super.__init__()
        self.lstm = LSTM()

    def training_step(self, batch, batch_idx):
        # TODO: Actually the batch and how we use them is different
        x, _ = batch
        x_hat = self.lstm(x)
        loss = F.mse_loss(x_hat, x)
        # TODO: Above needs to be changed
        return loss

    def validation_step(self, batch, batch_idx):  # TODO: Is this even correct? It doesn't return anything
        x, _ = batch
        x_hat = self.lstm(x)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)
        return loss

    def validation_epoch_end(self, val_step_outputs):
        for pred in val_step_outputs:
            print(pred)
            # do something with all the predictions from each validation_step

    # def configure_optimizer(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    #     return optimizer


# TODO: This is how we would train in the training script

# model
lstm = LSTMModule()

# train model
trainer = L.Trainer()
trainer.fit(model=lstm, train_dataloaders=train_loader)
