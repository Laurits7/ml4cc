import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*onnxscript\.values\.OnnxFunction\.param_schemas.*",
    category=FutureWarning,
    module=r"onnxscript\.converter",
)

import os
import glob
import hydra
from hydra.utils import instantiate
import shutil
import torch
import lightning as L
import awkward as ak
import torch.multiprocessing as mp
from omegaconf import DictConfig
from torch.utils.data import DataLoader, IterableDataset
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint
from ml4cc.tools.data import dataloaders as dl
from ml4cc.tools.evaluation import one_step as ose
from ml4cc.tools.evaluation import two_step as tse
from ml4cc.tools.evaluation import two_step_minimal as tsme

torch.set_float32_matmul_precision("medium")  # or 'high'


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def base_train(cfg: DictConfig, training_type: str):
    os.makedirs(cfg.training.log_dir, exist_ok=True)
    models_dir = (
        os.path.join(cfg.training.models_dir, training_type) if training_type != "one_step" else cfg.training.models_dir
    )
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=models_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=-1,
        save_weights_only=True,
        filename="{epoch:02d}-{val_loss:.2f}",
    )
    max_epochs = 2 if cfg.training.debug_run else cfg.training.trainer.max_epochs
    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=[
            TQDMProgressBar(refresh_rate=10),
            checkpoint_callback,
        ],
        logger=CSVLogger(cfg.training.log_dir, name=training_type),
        overfit_batches=1 if cfg.training.debug_run else 0,
    )
    return trainer, checkpoint_callback


def train_one_step(cfg: DictConfig, data_type: str):
    print(f"Training {cfg.models.one_step.model.name} for the one-step training.")
    model = instantiate(cfg.models.one_step.model)
    # datamodule = dl.OneStepDataModule(cfg=cfg, data_type=data_type, debug_run=cfg.training.debug_run)
    datamodule = dl.OneStepWindowedDataModule(cfg=cfg, data_type=data_type, debug_run=cfg.training.debug_run)
    if not cfg.training.model_evaluation_only:
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
    datamodule = dl.TwoStepPeakFindingDataModule(cfg=cfg, data_type=data_type, debug_run=cfg.training.debug_run)
    if not cfg.training.model_evaluation_only:
        trainer, checkpoint_callback = base_train(cfg, training_type="two_step_pf")
        trainer.fit(model=model, datamodule=datamodule)

        best_model_path = checkpoint_callback.best_model_path
        new_best_model_path = os.path.join(cfg.training.models_dir, "two_step_pf", "best_model.ckpt")
        shutil.copyfile(best_model_path, new_best_model_path)
        metrics_path = os.path.join(trainer.logger.log_dir, "metrics.csv")
    else:
        best_model_path = cfg.models.two_step.peak_finding.model.checkpoint.model
        metrics_path = cfg.models.two_step.peak_finding.model.checkpoint.losses
    return model, best_model_path, metrics_path


def train_two_step_clusterization(cfg: DictConfig, data_type: str):
    print(f"Training {cfg.models.two_step.clusterization.model.name} for the two-step clusterization training.")
    model = instantiate(cfg.models.two_step.clusterization.model)
    datamodule = dl.TwoStepClusterizationDataModule(cfg=cfg, data_type=data_type, debug_run=cfg.training.debug_run)
    if not cfg.training.model_evaluation_only:
        trainer, checkpoint_callback = base_train(cfg, training_type="two_step_cl")
        trainer.fit(model=model, datamodule=datamodule)

        best_model_path = checkpoint_callback.best_model_path
        new_best_model_path = os.path.join(cfg.training.models_dir, "two_step_cl", "best_model.ckpt")
        shutil.copyfile(best_model_path, new_best_model_path)
        metrics_path = os.path.join(trainer.logger.log_dir, "metrics.csv")
    else:
        best_model_path = cfg.models.two_step.clusterization.model.checkpoint.model
        metrics_path = cfg.models.two_step.clusterization.model.checkpoint.losses
    return model, best_model_path, metrics_path


def train_two_step_minimal(cfg: DictConfig, data_type: str):
    print(f"Training {cfg.models.two_step_minimal.model.name} for the two-step minimal training.")
    model = instantiate(cfg.models.two_step_minimal.model)
    datamodule = dl.TwoStepMinimalDataModule(cfg=cfg, data_type=data_type, debug_run=cfg.training.debug_run)
    if not cfg.training.model_evaluation_only:
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


def save_predictions(input_path: str, all_predictions: ak.Array, cfg: DictConfig, scenario: str):
    if not scenario == "two_step_cl":
        predictions_dir = cfg.training.predictions_dir
        base_scenario = "two_step" if "two_step" in scenario else scenario
        additional_dir_level = scenario if base_scenario == "two_step" else ""
        base_dir = cfg.dataset.data_dir
        original_dir = os.path.join(base_dir, base_scenario)
        predictions_dir = os.path.join(predictions_dir, additional_dir_level)
        os.makedirs(predictions_dir, exist_ok=True)
        output_path = input_path.replace(original_dir, predictions_dir)
        output_path = output_path.replace(".parquet", "_pred.parquet")
    else:
        output_path = input_path.replace("two_step_pf", "two_step_cl")

    input_data = ak.from_parquet(input_path)
    output_data = ak.copy(input_data)
    output_data["pred"] = ak.Array(all_predictions)  # pylint: disable=E1137
    print(f"Saving predictions to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ak.to_parquet(output_data, output_path, row_group_size=cfg.preprocessing.row_group_size)


def create_prediction_files(file_list: list, iterable_dataset: IterableDataset, model, cfg: DictConfig, scenario: str):
    num_files = 2 if cfg.training.debug_run else None
    with torch.no_grad():
        for path in file_list[:num_files]:
            dataset = dl.RowGroupDataset(path)
            iterable_dataset_ = iterable_dataset(dataset, device=DEVICE)
            dataloader = DataLoader(
                dataset=iterable_dataset_,
                batch_size=cfg.training.dataloader.batch_size,
                # num_workers=cfg.training.dataloader.num_dataloader_workers,
                # prefetch_factor=cfg.training.dataloader.prefetch_factor,
            )
            all_predictions = []
            for i, batch in enumerate(dataloader):
                predictions, _ = model(batch)

                all_predictions.append(predictions.detach().cpu().numpy())
            all_predictions = ak.concatenate(all_predictions, axis=0)
            save_predictions(input_path=path, all_predictions=all_predictions, cfg=cfg, scenario=scenario)


def evaluate_one_step(cfg: DictConfig, model, metrics_path: str) -> list:
    model.to(DEVICE)
    model.eval()
    dir_ = "*" if cfg.evaluation.training.eval_all_always else "test"
    wcp_path = os.path.join(cfg.dataset.data_dir, "one_step", dir_, "*")
    file_list = glob.glob(wcp_path)
    # iterable_dataset = dl.OneStepIterableDataset
    iterable_dataset = dl.OneStepWindowedIterableDataset

    # Create prediction files
    create_prediction_files(file_list, iterable_dataset=iterable_dataset, model=model, cfg=cfg, scenario="one_step")

    # Evaluate training
    ose.evaluate_training(cfg=cfg, metrics_path=metrics_path)


def evaluate_two_step_peak_finding(cfg: DictConfig, model, metrics_path: str) -> list:
    model.to(DEVICE)
    model.eval()
    wcp_path = os.path.join(cfg.dataset.data_dir, "two_step", "*", "*")
    file_list = glob.glob(wcp_path)
    iterable_dataset = dl.TwoStepPeakFindingIterableDataset
    # Create prediction files
    create_prediction_files(file_list, iterable_dataset=iterable_dataset, model=model, cfg=cfg, scenario="two_step_pf")

    # Evaluate training
    tse.evaluate_training(cfg=cfg, metrics_path=metrics_path, stage="peak_finding")


def evaluate_two_step_clusterization(cfg: DictConfig, model, metrics_path: str) -> list:
    model.to(DEVICE)
    model.eval()
    dir_ = "*" if cfg.evaluation.training.eval_all_always else "test"
    wcp_path = os.path.join(cfg.training.output_dir, "two_step", "predictions", "two_step_pf", dir_, "*")
    file_list = glob.glob(wcp_path)
    iterable_dataset = dl.TwoStepClusterizationIterableDataset

    # Create prediction files
    create_prediction_files(file_list, iterable_dataset=iterable_dataset, model=model, cfg=cfg, scenario="two_step_cl")

    # Evaluate training
    tse.evaluate_training(cfg=cfg, metrics_path=metrics_path, stage="clusterization")


def evaluate_two_step_minimal(cfg: DictConfig, model, metrics_path: str) -> list:
    model.to(DEVICE)
    model.eval()
    wcp_path = os.path.join(cfg.dataset.data_dir, "two_step", "*", "*")
    file_list = glob.glob(wcp_path)
    iterable_dataset = dl.TwoStepMinimalIterableDataset

    # Create prediction files
    create_prediction_files(file_list, iterable_dataset=iterable_dataset, model=model, cfg=cfg, scenario="two_step_min")

    # Evaluate training
    tsme.evaluate_training(cfg=cfg, metrics_path=metrics_path)


@hydra.main(config_path="../config", config_name="main.yaml", version_base=None)
def main(cfg: DictConfig):
    training_type = cfg.training.type
    if training_type == "one_step":
        print("Training one-step model.")
        model, best_model_path, metrics_path = train_one_step(cfg, data_type="")
        checkpoint = torch.load(best_model_path, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        evaluate_one_step(cfg, model, metrics_path)
    elif training_type == "two_step":
        model, best_model_path, metrics_path = train_two_step_peak_finding(cfg, data_type="")
        checkpoint = torch.load(best_model_path, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        evaluate_two_step_peak_finding(cfg, model, metrics_path)

        model, best_model_path, metrics_path = train_two_step_clusterization(cfg, data_type="")
        evaluate_two_step_clusterization(cfg, model, metrics_path)
    elif training_type == "two_step_minimal":
        model, best_model_path, metrics_path = train_two_step_minimal(cfg, data_type="")
        checkpoint = torch.load(best_model_path, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        evaluate_two_step_minimal(cfg, model, metrics_path)
    else:
        raise ValueError(f"Unknown training type: {training_type}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()  # pylint: disable=E1120
