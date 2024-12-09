import os
import hydra
import lightning as L
from ml4cc.models import LSTM
from omegaconf import DictConfig
from ml4cc.data.CEPC import dataloader as dl
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger


@hydra.main(config_path="../config", config_name="training.yaml", version_base=None)
def train(cfg: DictConfig):
    lstm = LSTM.LSTMModule()
    models_dir = os.path.join(cfg.training.output_dir, "models")
    log_dir = os.path.join(cfg.training.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    trainer = L.Trainer(
        max_epochs=cfg.training.trainer.max_epochs,
        callbacks=[
            TQDMProgressBar(refresh_rate=10),
            ModelCheckpoint(dirpath=models_dir, monitor="val_loss", mode="min"),
        ],
        logger=CSVLogger(log_dir, name="peakFinding")
    )
    datamodule = dl.CEPCDataModule(cfg=cfg, training_task="peakFinding", samples="all")
    trainer.fit(model=lstm, datamodule=datamodule)


if __name__ == "__main__":
    train()
