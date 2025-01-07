import os
import tqdm
import torch
import numpy as np
import pandas as pd
import awkward as ak
import mplhep as hep
import matplotlib.pyplot as plt
from ml4cc.tools.data import io
from ml4cc.tools.visualization import losses as l
from ml4cc.tools.visualization import regression as r
hep.style.use(hep.styles.CMS)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def filter_losses(metrics_path: str):
    metrics_data = pd.read_csv(metrics_path)
    val_loss = np.array(metrics_data['val_loss'])
    # train_loss = np.array(metrics_data['train_loss'])
    val_loss = val_loss[~np.isnan(val_loss)]
    # train_loss = train_loss[~np.isnan(train_loss)]
    return val_loss#, train_loss

def evaluate_training(model, dataloader, metrics_path, cfg):
    all_true = []
    all_preds = []
    true_save = []
    waveform_save = []
    prediction_save = []
    print("Prediction progress for TEST dataset")
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        wfs, true, wf_idx = batch
        pred = model(batch)
        waveform_save.append(np.concatenate(wfs.squeeze().detach().cpu().numpy()))
        prediction_save.append(pred.detach().cpu().numpy())
        true_save.append(true.detach().cpu().cpu().numpy())
        all_preds.extend(pred.detach().cpu().numpy())
        all_true.extend(true.detach().cpu().numpy())
    truth = np.array(all_true)
    preds = np.array(all_preds)

    # Save predictions file
    prediction_save = ak.Array(prediction_save)
    waveform_save = ak.Array(waveform_save)
    true_save = ak.Array(true_save)
    pred_file_data = ak.Array({
        "detected_peaks": prediction_save,
        "waveform": waveform_save,
        "target": true_save,
    })
    pred_file_path = os.path.join(cfg.training.output_dir, "predictions.parquet")
    io.save_array_to_file(data=pred_file_data, output_path=pred_file_path)

    # Save training results / metrics
    results_dir = os.path.join(cfg.training.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    val_loss = filter_losses(metrics_path)
    losses_output_path = os.path.join(results_dir, "losses.pdf")
    l.plot_loss_evolution(val_loss=val_loss, train_loss=None, output_path=losses_output_path)

    resolution_output_path = os.path.join(results_dir, "resolution.pdf")
    r.evaluate_resolution(truth, preds, output_path=resolution_output_path)

    distribution_output_path = os.path.join(results_dir, "true_pred_distributions.pdf")
    r.plot_true_pred_distributions(truth, preds, output_path=distribution_output_path)
