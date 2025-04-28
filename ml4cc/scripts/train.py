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
    os.makedirs(cfg.training.log_dir, exist_ok=True)
    os.makedirs(cfg.training.models_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.models_dir,
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
        logger=CSVLogger(cfg.training.log_dir, name=training_type)
    )
    return trainer, checkpoint_callback


def train_one_step(cfg: DictConfig, data_type: str):
    print(f"Training {cfg.models.one_step.model.name} for the one-step training.")
    model = instantiate(cfg.models.one_step.model)
    datamodule = dl.OneStepDataModule(cfg=cfg, data_type=data_type)
    if not cfg.model_evaluation_only:
        trainer, checkpoint_callback = base_train(cfg, training_type="one_step")
        trainer.fit(model=model, datamodule=datamodule)

        best_model_path = checkpoint_callback.best_model_path
        new_best_model_path = os.path.join(cfg.training.models_dir, "best_model.ckpt")
        shutil.copyfile(best_model_path, new_best_model_path)
        metrics_path = os.path.join(trainer.logger.log_dir, "metrics.csv")
    else:
        best_model_path = cfg.models.one_step.model.checkpoint.model
        metrics_path = cfg.models.one_step.model.checkpoint.losses
    return model, best_model_path, metrics_path


def train_two_step_peak_finding(cfg: DictConfig, data_type: str):
    print(f"Training {cfg.models.two_step.peak_finding.model.name} for the two-step peak-finding training.")
    model = instantiate(cfg.models.two_step.peak_finding.model)
    datamodule = dl.TwoStepPeakFindingDataModule(cfg=cfg, data_type=data_type)
    if not cfg.model_evaluation_only:
        trainer, checkpoint_callback = base_train(cfg, training_type="two_step_peak_finding")
        trainer.fit(model=model, datamodule=datamodule)

        best_model_path = checkpoint_callback.best_model_path
        new_best_model_path = os.path.join(cfg.training.models_dir, "best_model.ckpt")
        shutil.copyfile(best_model_path, new_best_model_path)
        metrics_path = os.path.join(trainer.logger.log_dir, "metrics.csv")
    else:
        best_model_path = cfg.models.two_step.peak_finding.model.checkpoint.model
        metrics_path = cfg.models.two_step.peak_finding.model.checkpoint.losses
    return model, best_model_path, metrics_path


def train_two_step_clusterization(cfg: DictConfig, data_type: str):
    print(f"Training {cfg.models.two_step.clusterization.model.name} for the two-step clusterization training.")
    model = instantiate(cfg.models.two_step.clusterization.model)
    datamodule = dl.TwoStepClusterizationDataModule(cfg=cfg, data_type=data_type)
    if not cfg.model_evaluation_only:
        trainer, checkpoint_callback = base_train(cfg, training_type="two_step_clusterization")
        trainer.fit(model=model, datamodule=datamodule)

        best_model_path = checkpoint_callback.best_model_path
        new_best_model_path = os.path.join(cfg.training.models_dir, "best_model.ckpt")
        shutil.copyfile(best_model_path, new_best_model_path)
        metrics_path = os.path.join(trainer.logger.log_dir, "metrics.csv")
    else:
        best_model_path = cfg.models.two_step.clusterization.model.checkpoint.model
        metrics_path = cfg.models.two_step.clusterization.model.checkpoint.losses
    return model, best_model_path, metrics_path


def train_two_step_minimal(cfg: DictConfig, data_type: str):
    print(f"Training {cfg.models.two_step_minimal.model.name} for the two-step minimal training.")
    model = instantiate(cfg.models.two_step_minimal.model)
    datamodule = dl.TwoStepMinimalDataModule(cfg=cfg, data_type=data_type)
    if not cfg.model_evaluation_only:
        trainer, checkpoint_callback = base_train(cfg, training_type="two_step_minimal")
        trainer.fit(model=model, datamodule=datamodule)

        best_model_path = checkpoint_callback.best_model_path
        new_best_model_path = os.path.join(cfg.training.models_dir, "best_model.ckpt")
        shutil.copyfile(best_model_path, new_best_model_path)
        metrics_path = os.path.join(trainer.logger.log_dir, "metrics.csv")
    else:
        best_model_path = cfg.models.two_step_minimal.model.checkpoint.model
        metrics_path = cfg.models.two_step_minimal.model.checkpoint.losses
    return model, best_model_path, metrics_path


def get_CEPC_evaluation_scenarios(cfg: DictConfig):
    evaluation_scenarios = []
    for pid in cfg.dataset.particle_types:
        for energy in cfg.dataset.particle_energies:
            scenario_name = f"{pid}_{energy}"
            evaluation_scenarios.append(scenario_name)
    return evaluation_scenarios


def get_FCC_evaluation_scenarios(cfg: DictConfig) -> list:
    evaluation_scenarios = []
    for energy in cfg.dataset.particle_energies:
        scenario_name = f"{energy}"
        evaluation_scenarios.append(scenario_name)
    return evaluation_scenarios


def evaluate_one_step(cfg: DictConfig) -> list:
    pass


def evaluate_two_step_peak_finding(cfg: DictConfig):
    pass
    # TODO: Evaluate training
    # TODO: Create prediction files for the clusterization step


def evaluate_two_step_clusterization(cfg: DictConfig):
    pass


def evaluate_two_step_minimal(cfg: DictConfig):
    pass


@hydra.main(config_path="../config", config_name="main.yaml", version_base=None)
def main(cfg: DictConfig):
    training_type = cfg.training.type

    if training_type == "one_step":
        model, best_model_path, metrics_path = train_one_step(cfg, data_type="")
        evaluate_one_step(cfg, model, best_model_path, metrics_path)
    elif training_type == "two_step":
        model, best_model_path, metrics_path = train_two_step_peak_finding(cfg, data_type="")
        evaluate_two_step_peak_finding(cfg, model, best_model_path, metrics_path, data_type="")

        model, best_model_path, metrics_path = train_two_step_clusterization(cfg)
        evaluate_two_step_clusterization(cfg, model, best_model_path, metrics_path, data_type="")
    elif training_type == "two_step_minimal":
        model, best_model_path, metrics_path = train_two_step_minimal(cfg, data_type="")
        evaluate_two_step_minimal(cfg, model, best_model_path, metrics_path)
    else:
        raise ValueError(f"Unknown training type: {training_type}")



if __name__ == "__main__":
    main()