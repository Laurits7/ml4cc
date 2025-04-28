import os
import glob
import hydra
import shutil
import torch
import lightning as L
from ml4cc.models import transformer
from omegaconf import DictConfig
from ml4cc.data.CEPC import dataloader as cdl
from ml4cc.data.FCC import dataloader as fdl
from ml4cc.tools.evaluation import one_step
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint


@hydra.main(config_path="../config", config_name="training.yaml", version_base=None)
def train(cfg: DictConfig):
    input_dim = cfg.datasets[cfg.training.data.dataset].input_dim
    model = transformer.TransformerModule(input_dim=input_dim)
    if not cfg.model_evaluation_only:
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
        if cfg.training.data.dataset == "CEPC":
            datamodule = cdl.OneStepCEPCDataModule(cfg=cfg)
        else:
            datamodule = fdl.OneStepFCCDataModule(cfg=cfg)
        trainer.fit(model=model, datamodule=datamodule)

        # Get best model
        best_model_path = checkpoint_callback.best_model_path
        new_best_model_path = os.path.join(models_dir, "best_model.ckpt")
        shutil.copyfile(best_model_path, new_best_model_path)
        metrics_path = os.path.join(trainer.logger.log_dir, "metrics.csv")
    else:
        best_model_path = cfg.models.one_step.transformer.checkpoint.model  # TODO: Change for different models
        metrics_path = cfg.models.one_step.transformer.checkpoint.losses  # TODO: Change for different models

    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    if cfg.training.data.dataset == "CEPC":
        data_paths = os.path.join(cfg.datasets.CEPC.data_dir, 'peakFinding', 'test')
        dataset = cdl.CEPCDataset(data_path=data_paths)
        test_dataset = cdl.OneStepIterableDataSet(
            dataset=dataset,
            cfg=cfg,
            dataset_type="test",
        )
        test_loader = DataLoader(test_dataset)
    else:
        data_paths = list(glob.glob(os.path.join(cfg.datasets.FCC.data_dir, 'test', "*.parquet")))
        test_dataset = fdl.OneStepIterableDataSet(
            data_paths=data_paths,
            cfg=cfg,
            dataset_type="test",
        )
        test_loader = DataLoader(test_dataset)
    one_step.evaluate_training(
        model=model,
        dataloader=test_loader,
        metrics_path=metrics_path,
        cfg=cfg
    )


if __name__ == "__main__":
    train()
