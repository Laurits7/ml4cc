import numpy as np
import mplhep as hep
import boost_histogram as bh
import matplotlib.pyplot as plt
hep.style.use(hep.styles.CMS)


def calculate_resolution(truth: np.array, preds: np.array) -> np.array:
    """ We calculate the resolution as IQR/median as stdev is more affected by outliers"""
    ratios = truth/preds
    iqr = np.quantile(ratios, 0.75) - np.quantile(ratios, 0.25)
    median = np.quantile(ratios, 0.5)
    resolution = iqr / median
    return resolution, ratios

def to_bh(data, bins, cumulative=False):
    h1 = bh.Histogram(bh.axis.Variable(bins))
    h1.fill(data)
    if cumulative:
        h1[:] = np.sum(h1.values()) - np.cumsum(h1)
    return h1


def evaluate_resolution(truth: np.array, preds: np.array, output_path: str) -> None:
    resolution, ratios = calculate_resolution(truth, preds)
    bins = np.linspace(0.5, 1.5, 101)
    hep.histplot(to_bh(ratios, bins=bins), ax=plt.gca(), density=True)
    plt.axvline(x=1, ls='--')
    plt.figtext(0.5, 0.5, f'IQR={resolution:.4f}')
    plt.xlabel(r"$n_{cls}^{true}/n_{cls}^{pred}$")
    plt.close("all")


def plot_true_pred_distributions(truth: np.array, preds: np.array, output_path: str) -> None:
    bins = np.linspace(
        start=np.min(np.concatenate((truth, preds))),
        stop=np.max(np.concatenate((truth, preds))),
        num=25
    )
    plt.hist(preds, bins=bins, histtype='step', label="Reconstructed", density=True)
    plt.hist(truth, bins=bins, histtype='step', label="Target", density=True)
    plt.xlabel("Number of primary clusters")
    plt.ylabel("Number entries [a.u.]")
    plt.legend()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')
