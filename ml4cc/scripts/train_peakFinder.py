import hydra
import lightning as L
from ml4cc.models import LSTM
from omegaconf import DictConfig
from ml4cc.tools.callbacks import progressbar as pb
from ml4cc.data.CEPC import dataloader as dl


@hydra.main(config_path="../config", config_name="training.yaml", version_base=None)
def train(cfg: DictConfig):
    lstm = LSTM.LSTMModule()
    bar = pb.ProgressBar()
    trainer = L.Trainer(max_epochs=cfg.training.trainer.max_epochs, callbacks=[bar])
    datamodule = dl.CEPCDataModule(cfg=cfg, training_task="peakFinding", samples="all")
    trainer.fit(model=lstm, datamodule=datamodule)


if __name__ == "__main__":
    train()
