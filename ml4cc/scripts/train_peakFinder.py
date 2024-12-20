import os
import hydra
import shutil
import lightning as L
from ml4cc.models import LSTM
from omegaconf import DictConfig
from ml4cc.data.CEPC import dataloader as dl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint


@hydra.main(config_path="../config", config_name="training.yaml", version_base=None)
def train(cfg: DictConfig):
    lstm = LSTM.LSTMModule()
    models_dir = os.path.join(cfg.training.output_dir, "models")
    log_dir = os.path.join(cfg.training.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=models_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=-1,
        save_weights_only=True,
        filename="peakFinding-{epoch:02d}-{val_loss:.2f}"
    )
    trainer = L.Trainer(
        max_epochs=cfg.training.trainer.max_epochs,
        callbacks=[
            TQDMProgressBar(refresh_rate=10),
            checkpoint_callback,
        ],
        logger=CSVLogger(log_dir, name="peakFinding")
    )
    datamodule = dl.CEPCDataModule(cfg=cfg, training_task="peakFinding", samples="all")
    trainer.fit(model=lstm, datamodule=datamodule)

    # Get best model
    best_model_path = checkpoint_callback.best_model_path
    new_best_model_path = os.path.join(models_dir, "best_model.ckpt")
    shutil.copyfile(best_model_path, new_best_model_path)

    # best_model = lstm.load_from_checkpoint(best_model_path)
    # trainer.test(model=lstm, datamodule=datamodule, ckpt_path='best')

    # Need to write to new .parquet file?



if __name__ == "__main__":
    train()
