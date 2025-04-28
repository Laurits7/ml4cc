import os
import hydra
from hydra.utils import instantiate
import shutil
import torch
import lightning as L
from omegaconf import DictConfig
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint
from ml4cc.tools.data import dataloaders as dl


def base_train(cfg: DictConfig, training_type: str):
    if not cfg.model_evaluation_only:
        output_dir = os.path.join(cfg.training.output_dir, training_type)
        models_dir = os.path.join(output_dir, "models")
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=models_dir,
            monitor="val_loss",
            mode="min",
            save_top_k=-1,
            save_weights_only=True,
            filename="{epoch:02d}-{val_loss:.2f}"
        )
        trainer = L.Trainer(
            max_epochs=cfg.training.trainer.max_epochs,
            callbacks=[
                TQDMProgressBar(refresh_rate=10),
                checkpoint_callback,
            ],
            logger=CSVLogger(log_dir, name=training_type)
        )
    return trainer


def train_one_step(cfg: DictConfig):
    print(f"Training {cfg.models.one_step.model.name} for the one-step training.")
    model = instantiate(cfg.models.one_step.model)
    trainer = base_train(cfg, training_type="one_step")
    datamodule = dl.OneStepDataModule(cfg=cfg, data_type="K")
    trainer.fit(model=model, datamodule=datamodule)


def train_two_step_peak_finding(cfg: DictConfig):
    print(f"Training {cfg.models.two_step.peak_finding.model.name} for the two-step peak-finding training.")
    model = instantiate(cfg.models.two_step.peak_finding.model)
    trainer = base_train(cfg, training_type="two_step_peak_finding")
    datamodule = dl.TwoStepPeakFindingDataModule(cfg=cfg, data_type="K")
    trainer.fit(model=model, datamodule=datamodule)


def train_two_step_clusterization(cfg: DictConfig):
    print(f"Training {cfg.models.two_step.clusterization.model.name} for the two-step clusterization training.")
    model = instantiate(cfg.models.two_step.clusterization.model)
    trainer = base_train(cfg, training_type="two_step_clusterization")
    datamodule = dl.TwoStepClusterizationDataModule(cfg=cfg, data_type="K")
    trainer.fit(model=model, datamodule=datamodule)


def train_two_step_minimal(cfg: DictConfig):
    print(f"Training {cfg.models.two_step_minimal.model.name} for the two-step minimal training.")
    model = instantiate(cfg.models.two_step_minimal.model)
    trainer = base_train(cfg, training_type="two_step_minimal")
    datamodule = dl.TwoStepMinimalDataModule(cfg=cfg, data_type="K")
    trainer.fit(model=model, datamodule=datamodule)


@hydra.main(config_path="../config", config_name="main.yaml", version_base=None)
def main(cfg: DictConfig):
    training_type = cfg.training.type

    if training_type == "one_step":
        train_one_step(cfg)
    elif training_type == "two_step":
        train_two_step_peak_finding(cfg)
        train_two_step_clusterization(cfg)
    elif training_type == "two_step_minimal":
        train_two_step_minimal(cfg)
    else:
        raise ValueError(f"Unknown training type: {training_type}")



if __name__ == "__main__":
    main()