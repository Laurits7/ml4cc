import math
import torch
import torch.nn as nn
import lightning as L
import torch.optim as optim
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)


class WaveFormTransformer(nn.Module):
    def __init__(
            self,
            input_dim: int, # 1024 or 3000
            d_model: int, # 512
            num_heads: int, # 16
            num_layers: int, # 3
            hidden_dim: int, # 4*d_model
            num_classes: int, # 1, either bkg or signal
            max_len: int, # As we have fixed nr, then it max_len=input_dim
            dropout: float=0.1
    ):
        super().__init__()
        self.input_projection = nn.Linear(1, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.peak_finding_classifier = nn.Linear(d_model, num_classes)
        # self.clusterizer = nn.Linear(d_model, 1)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.layernorm(x)
        x = self.peak_finding_classifier(x)  # Shape: [batch_size, seq_length, num_classes]
        x = F.relu(x)
        x = x.sum(dim=1)  # Shape: [batch_size, num_classes]
        # x = self.clusterizer(x)
        return x


class TransformerModule(L.LightningModule):
    def __init__(self, input_dim):
        super().__init__()
        self.transformer = WaveFormTransformer(
            input_dim=input_dim,
            d_model=512,
            num_heads=8,
            num_layers=3,
            hidden_dim=4*512,
            num_classes=1,
            max_len=input_dim
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
        predicted_labels, _ = self.forward(batch)
        return predicted_labels

    def test_step(self, batch, batch_idx):
        predicted_labels, _ = self.forward(batch)
        return predicted_labels

    def forward(self, batch):
        waveform, target = batch
        predicted_labels = self.transformer(waveform).squeeze()
        return predicted_labels, target
