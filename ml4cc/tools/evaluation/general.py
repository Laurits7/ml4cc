import os
import glob
import warnings
import numpy as np
import pandas as pd
import awkward as ak
from omegaconf import DictConfig
from ml4cc.tools.visualization import losses as l



def filter_losses(metrics_path: str):
    metrics_data = pd.read_csv(metrics_path)
    val_loss = np.array(metrics_data["val_loss"])
    # train_loss = np.array(metrics_data['train_loss'])
    val_loss = val_loss[~np.isnan(val_loss)]
    # train_loss = train_loss[~np.isnan(train_loss)]
    return val_loss  # , train_loss


def collect_all_results(predictions_dir: str, cfg: DictConfig) -> dict:
    """
    Collect all results from the predictions directory.

    Parameters:
        predictions_dir (str): Path to the predictions directory.
        cfg (DictConfig): Configuration.

    Returns:
        dict: Dictionary containing results for each.
    """
    results = {}
    energies = cfg.dataset.particle_energies
    particle_types = cfg.dataset.particle_types
    for pid in particle_types:
        results[pid] = {}
        for energy in energies:
            file_name = f"{energy}_1_pred.parquet" if cfg.dataset.name == "FCC" else f"signal_{pid}_{energy}_*.parquet"
            pid_energy_wcp = os.path.join(predictions_dir, "test", file_name)
            pid_energy_files = list(glob.glob(pid_energy_wcp))
            if len(pid_energy_files) == 0:
                warnings.warn(f"No prediction files found for {pid} at {energy} GeV in {predictions_dir}.")
                continue
            pid_energy_true = []
            pid_energy_pred = []
            for pid_energy_file in pid_energy_files:
                data = ak.from_parquet(pid_energy_file)
                pid_energy_true.append(data["target"])
                pid_energy_pred.append(data["pred"])
            pid_energy_pred = ak.concatenate(pid_energy_pred, axis=0)
            pid_energy_true = ak.concatenate(pid_energy_true, axis=0)
            results[pid][energy] = {"true": pid_energy_true, "pred": pid_energy_pred}
    return results


def evaluate_losses(
    metrics_path: str, model_name: str = "", loss_name: str = "BCE", results_dir: str = ""
):
    # Visualize losses for the training.
    losses = filter_losses(metrics_path=metrics_path)
    losses_output_path = os.path.join(results_dir, "losses.png")

    lp = l.LossesMultiPlot(loss_name=loss_name)
    loss_results = {model_name: {"val_loss": losses}}
    lp.plot_algorithms(results=loss_results, output_path=losses_output_path)
