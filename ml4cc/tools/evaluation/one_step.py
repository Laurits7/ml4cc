import os
import tqdm
import pandas as pd
import numpy as np
from ml4cc.tools.data import io
from ml4cc.tools.visualization import losses as l
from ml4cc.tools.visualization import classification as cl



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
    all_waveforms = []
    print("Prediction progress for TEST dataset")
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        waveform, target = batch
        pred = model(batch)[0]
        all_true.append(target.detach().cpu().numpy())
        all_preds.append(pred.detach().cpu().numpy())
        all_waveforms.append(waveform.detach().cpu().numpy())
    pred_file_data = ak.Array({
        "detected_peaks": all_preds,
        "waveform": all_waveforms,
        "target": all_true,
    })
    results_dir = os.path.join(cfg.training.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    pred_file_path = os.path.join(results_dir, "predictions.parquet")
    io.save_array_to_file(data=pred_file_data, output_path=pred_file_path)

    cl.plot_classification(truth=all_true, preds=all_preds, output_dir=results_dir)

    # val_loss, train_loss = filter_losses(metrics_path)
    val_loss = filter_losses(metrics_path)
    losses_output_path = os.path.join(output_dir, "losses.png")
    l.plot_loss_evolution(val_loss=val_loss, train_loss=None, output_path=losses_output_path)