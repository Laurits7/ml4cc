import os
import tqdm
import numpy as np
import pandas as pd
import awkward as ak
import mplhep as hep
import matplotlib.pyplot as plt
from ml4cc.tools.data import io
from ml4cc.tools.visualization import losses as l
from ml4cc.tools.visualization import classification as cl
hep.style.use(hep.styles.CMS)


def filter_losses(metrics_path: str):
    metrics_data = pd.read_csv(metrics_path)
    val_loss = np.array(metrics_data['val_loss'])
    train_loss = np.array(metrics_data['train_loss'])
    val_loss = val_loss[~np.isnan(val_loss)]
    train_loss = train_loss[~np.isnan(train_loss)]
    return val_loss, train_loss


def create_pred_values(preds: np.array, cfg):
    window_size = 15  # from cfg
    pred_vector = []
    zero_count = int((window_size - 1) / 2)
    for pred in preds:
        if pred > 0.5:  # If detected peak
            pred_vector.extend([0]*zero_count + [1] + [0]*zero_count)  # So the middle of the window is the location of the peak
        else:
            pred_vector.extend([0]*zero_count)
    return np.array(pred_vector)


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
        prediction_save.append(create_pred_values(pred.detach().numpy(), cfg))
        waveform_save.append(np.concatenate(wfs.squeeze().detach().numpy()))
        true_save.append(np.concatenate(true.squeeze().detach().numpy()))
        all_preds.extend(pred.detach().numpy())
        all_true.extend(true.detach().numpy())
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
    output_dir = os.path.join(cfg.training.output_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    cl.plot_classification(truth=truth, preds=preds, output_dir=output_dir)

    val_loss, train_loss = filter_losses(metrics_path)
    losses_output_path = os.path.join(output_dir, "losses.pdf")
    l.plot_loss_evolution(val_loss=val_loss, train_loss=train_loss, output_path=losses_output_path)


def plot_prediction_v_true(wfs, vals, window_size=15):
    wf = np.concatenate(wfs.squeeze().numpy())
    plt.plot(np.arange(len(wf)), wf)
    for i, x in enumerate(vals):
        loc = (window_size/2) + i * window_size
        if x > 0.5:
            plt.axvline(loc, ymax=4, linestyle='--', color='red')
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    plt.grid()
