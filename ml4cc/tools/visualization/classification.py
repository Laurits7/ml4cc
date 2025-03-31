import os
import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
hep.style.use(hep.styles.CMS)


def plot_roc_curve(truth: np.array, preds: np.array, output_path: str='') -> None:
    """ Plots the ROC curve based on the true and predicted values

    Parameters:
        truth : np.array
            True labels in the waveform.
        preds : np.array
            Predicted labels in the waveform.
        output_path : str
            [default: ''] Path where figure will be saved. If empty string, figure will not be saved

    Returns:
        None
    """
    fpr, tpr, thresholds = roc_curve(truth, preds)
    auc_value = roc_auc_score(truth, preds)
    plt.plot(fpr, tpr)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.figtext(0.5, 0.5, f'AUC={auc_value:.4f}')
    if output_path != '':
        plt.savefig(output_path, bbox_inches='tight')
    plt.close("all")


def plot_classifier_scores_distribution(truth: np.array, preds: np.array, output_path: str='') -> None:
    """ Plots the distribution of classifier scores based on the true and predicted values

    Parameters:
        truth : np.array
            True labels in the waveform.
        preds : np.array
            Predicted labels in the waveform.
        output_path : str
            [default: ''] Path where figure will be saved. If empty string, figure will not be saved

    Returns:
        None
    """
    bkg_idx = truth == 0
    sig_idx = truth == 1
    bins = np.linspace(0, 1, num=25)
    plt.hist(preds[bkg_idx], color='r', label='BKG', histtype='step', density=True, bins=bins)
    plt.hist(preds[sig_idx], color='b', label='SIG', histtype='step', density=True, bins=bins)
    plt.legend()
    plt.xlabel(r"$\mathcal{D}_p$")
    plt.ylabel("Count [a.u.]")
    if output_path != '':
        plt.savefig(output_path)
    plt.close("all")


def plot_classification(truth: np.array, preds: np.array, output_dir: str='') -> None:
    """ Plots the distribution of classifier scores based on the true and predicted values

    Parameters:
        truth : np.array
            True labels in the waveform.
        preds : np.array
            Predicted labels in the waveform.
        output_dir : str
            Directory where output plots will be saved

    Returns:
        None
    """
    roc_output_path = os.path.join(output_dir, "roc.png")
    plot_roc_curve(truth=truth, preds=preds, output_path=roc_output_path)
    classifier_score_path = os.path.join(output_dir, "cls_scores.png")
    plot_classifier_scores_distribution(truth=truth, preds=preds, output_path=classifier_score_path)
