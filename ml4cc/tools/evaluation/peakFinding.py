import os
import tqdm
import numpy as np
import pandas as pd
from ml4cc.tools.visualization import losses as l
from ml4cc.tools.visualization import classification as cl


def filter_losses(metrics_path: str):
    metrics_data = pd.read_csv(metrics_path)
    val_loss = np.array(metrics_data['val_loss'])
    train_loss = np.array(metrics_data['train_loss'])
    val_loss = val_loss[~np.isnan(val_loss)]
    train_loss = train_loss[~np.isnan(train_loss)]
    return val_loss, train_loss


def evaluate_training(model, dataloader, metrics_path, cfg):
    all_true = []
    all_preds = []
    print("Prediction progress for TEST dataset")
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        true = batch[1]
        pred = model(batch)
        all_preds.extend(pred.detach().numpy())
        all_true.extend(true.detach().numpy())
    truth = np.array(all_true)
    preds = np.array(all_preds)

    output_dir = os.path.join(cfg.training.output_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    cl.plot_classification(truth=truth, preds=preds, output_dir=output_dir)

    val_loss, train_loss = filter_losses(metrics_path)
    l.plot_loss_evolution(val_loss=val_loss, train_loss=train_loss)
