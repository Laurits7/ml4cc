import hydra
import lightning as L
from ml4cc.models import DGCNN
from omegaconf import DictConfig
from ml4cc.data.CEPC import dataloader as dl
from lightning.pytorch.callbacks import TQDMProgressBar


@hydra.main(config_path="../config", config_name="training.yaml", version_base=None)
def train(cfg: DictConfig):
    dgcnn = DGCNN.DGCNNModule()  # TODO: Implement + check what target value should there be?
    trainer = L.Trainer(max_epochs=cfg.training.trainer.max_epochs, callbacks=[TQDMProgressBar(refresh_rate=10)])
    datamodule = dl.CEPCDataModule(cfg=cfg, training_task="peakFinding", samples="all")
    trainer.fit(model=lstm, datamodule=datamodule)


if __name__ == "__main__":
    train()
