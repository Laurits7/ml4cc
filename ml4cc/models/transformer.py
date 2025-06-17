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
        return x + self.pe[:, : x.size(1), :].to(x.device)


class WaveFormTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,  # 1024 or 3000
        d_model: int,  # 512
        num_heads: int,  # 16
        num_layers: int,  # 3
        hidden_dim: int,  # 4*d_model
        num_classes: int,  # 1, either bkg or signal
        max_len: int,  # As we have fixed nr, then it max_len=input_dim
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.peak_finding_classifier = nn.Linear(d_model, num_classes)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x = (x - mean) / (std + 1e-6)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        # x = self.layernorm(x)
        # Shape: [batch_size, seq_length, num_classes]
        x = self.peak_finding_classifier(x)
        x = x.sum(dim=1)  # Shape: [batch_size, num_classes]
        x = F.softplus(x)
        return x


class TransformerModule(L.LightningModule):
    def __init__(self, name: str, hyperparameters: dict, checkpoint: dict = None):
        super().__init__()
        self.name = name
        self.hyperparameters = hyperparameters
        self.checkpoint = checkpoint
        self.transformer = WaveFormTransformer(
            input_dim=15, # self.hyperparameters["input_dim"], If windowed, then 15, otherwise 1
            d_model=self.hyperparameters["d_model"],
            num_heads=self.hyperparameters["num_heads"],
            num_layers=self.hyperparameters["num_layers"],
            hidden_dim=self.hyperparameters["hidden_dim"],
            num_classes=self.hyperparameters["num_classes"],
            max_len=self.hyperparameters["max_len"],
        )
        self.lr = self.hyperparameters["lr"]

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
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5),
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
        waveform, target = batch
        predicted_labels = self.transformer(waveform).squeeze()
        print("PREDICTED:", predicted_labels)
        print("TARGET:", target)
        return predicted_labels, target

    def on_after_backward(self):
        total_norm = 0
        for name, p in self.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f"Gradient norm: {total_norm:.6f}")