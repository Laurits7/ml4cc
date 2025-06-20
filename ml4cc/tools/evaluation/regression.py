import numpy as np
import awkward as ak


def calculate_resolution(truth: np.array, preds: np.array) -> np.array:
    """We calculate the resolution as IQR/median as stdev is more affected by outliers"""
    ratios = truth / preds
    iqr = np.quantile(ratios, 0.75) - np.quantile(ratios, 0.25)
    median = np.quantile(ratios, 0.5)
    resolution = iqr / median
    return resolution, ratios


def collect_resolution_results(results: dict) -> dict:
    """ Collects the resolution results from different energies """
    true = ak.sum(results["true"] == 1, axis=-1)
    pred = results["pred"]
    resolution, ratios = calculate_resolution(true, pred)
    resolution_results = {
        "resolution": resolution,
        "ratios": ratios,
        "true": true,
        "pred": pred
    }
    return resolution_results


def get_per_energy_metrics(results):
    per_energy_metrics = {}
    for pid, pid_results in results.items():
        per_energy_metrics[pid] = {}
        for energy, energy_results in pid_results.items():
            per_energy_metrics[pid][energy] = collect_resolution_results(energy_results)
    return per_energy_metrics
