import torch
import lightning as L
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DNNModel(nn.Module):
    def __init__(self, nfeature):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(nfeature, 32)
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
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.output(x)
        return x


class RNNModel(nn.Module):
    def __init__(self, nfeature):
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
    def __init__(self, lr, model_, n_features):
        super().__init__()
        self.lr = lr
        self.model = model_(n_features)

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
        # return optim.RMSprop(self.parameters(), lr=0.001)  # They use this
        return optim.AdamW(self.parameters(), lr=0.001)

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
