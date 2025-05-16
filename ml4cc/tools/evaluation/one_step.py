import os
import glob
import awkward as ak
from ml4cc.tools.data import io
from ml4cc.tools.visualization import losses as l
from ml4cc.tools.visualization import regression as r
from ml4cc.tools.evaluation import general as g


def evaluate_training(cfg, metrics_path):
    results_dir = cfg.training.results_dir
    os.makedirs(results_dir, exist_ok=True)
    predictions_dir = cfg.training.predictions_dir
    test_dir = os.path.join(predictions_dir, "test")
    test_wcp = glob.glob(os.path.join(test_dir, "*"))
    all_true = []
    all_preds = []
    for path in test_wcp:
        data = ak.from_parquet(path)
        all_true.append(ak.sum(data.target, axis=-1))
        all_preds.append(data.pred)

    truth = ak.flatten(all_true, axis=None)
    preds = ak.flatten(all_preds, axis=None)

    resolution_output_path = os.path.join(results_dir, "resolution.pdf")
    r.evaluate_resolution(truth, preds, output_path=resolution_output_path)

    distribution_output_path = os.path.join(results_dir, "true_pred_distributions.pdf")
    r.plot_true_pred_distributions(truth, preds, output_path=distribution_output_path)

    val_loss = g.filter_losses(metrics_path)
    losses_output_path = os.path.join(cfg.training.output_dir, "losses.png")
    l.plot_loss_evolution(val_loss=val_loss, train_loss=None, output_path=losses_output_path)
