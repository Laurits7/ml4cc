import os
import hydra
import shutil
from ml4cc.tools.evaluation import clusterization as cl
import lightning as L
from ml4cc.models import DGCNN
from ml4cc.models import simpler_models as sm
from omegaconf import DictConfig
from ml4cc.data.CEPC import dataloader as cdl
from ml4cc.data.FCC import dataloader as fdl
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint


@hydra.main(config_path="../config", config_name="training.yaml", version_base=None)
def train(cfg: DictConfig):
    # model = DGCNN.DGCNNModule(cfg.models.cluster_counting.DGCNN.hyperparameters)
    model = sm.SimplerModelModule(lr=0.0001, model_=sm.RNNModel, n_features=3000) #For DNN it is 0.001
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
            filename="clusterization-{epoch:02d}-{val_loss:.2f}"
        )
        trainer = L.Trainer(
            max_epochs=cfg.training.trainer.max_epochs,
            callbacks=[
                TQDMProgressBar(refresh_rate=10),
                checkpoint_callback,
            ],
            logger=CSVLogger(log_dir, name="clusterization")
        )
        if cfg.training.data.dataset == "CEPC":
            datamodule = cdl.ClusterizationCEPCDataModule(cfg=cfg)
        else:  # TODO: Implement
            datamodule = fdl.ClusterizationFCCDataModule(cfg=cfg)
        trainer.fit(model=model, datamodule=datamodule)

        # Get best model
        best_model_path = checkpoint_callback.best_model_path
        new_best_model_path = os.path.join(models_dir, "best_model.ckpt")
        shutil.copyfile(best_model_path, new_best_model_path)
        metrics_path = os.path.join(trainer.logger.log_dir, "metrics.csv")
    else:
        best_model_path = cfg.checkpoints.clusterization.DGCNN.model
        metrics_path = cfg.checkpoints.clusterization.DGCNN.losses

    # model = sm.SimplerModelModule(lr=0.0001, model_=sm.DNNModel, n_features=3000)
    # DGCNN.DGCNNModule.load_from_checkpoint(best_model_path, weights_only=True)
    model.eval()
    if cfg.training.data.dataset == "CEPC":
        # TODO: for sample in samples:
        sample = 'kaon'
        data_dir = os.path.join(cfg.datasets.CEPC.data_dir, 'clusterization', 'test', sample)
        dataset = cdl.CEPCDataset(data_path=data_dir)
        test_dataset = cdl.ClusterizationIterableDataSet(
            dataset=dataset,
            cfg=cfg,
            dataset_type="test",
        )
        test_loader = DataLoader(test_dataset, batch_size=200)
    else:  # TODO: Implement.
        data_dir = os.path.join(cfg.datasets.FCC.data_dir, 'test')
        test_dataset = fdl.IterableFCCDataset(
            data_path=data_dir,
            cfg=cfg,
            dataset_type="test",
        )
        test_loader = DataLoader(test_dataset, batch_size=200)

    cl.evaluate_training(model, test_loader, metrics_path, cfg)
    # TODO: Write evalution part - plotting distribution, losses, etc.



if __name__ == "__main__":
    train()
