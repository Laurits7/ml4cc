import numpy as np
import awkward as ak


def calculate_resolution(truth: np.array, preds: np.array) -> np.array:
    """We calculate the resolution as IQR/median as stdev is more affected by outliers"""
    ratios = truth / preds
    iqr = np.quantile(ratios, 0.75) - np.quantile(ratios, 0.25)
    median = np.quantile(ratios, 0.5)
    resolution = iqr / median
    return resolution, median, ratios


def collect_resolution_results(results: dict) -> dict:
    """Collects the resolution results from different energies"""
    resolution, median, ratios = calculate_resolution(results["true"], results["pred"])
    resolution_results = {
        "resolution": resolution,
        "median": median,
        "ratios": ratios,
        "true": results["true"],
        "pred": results["pred"]
    }
    return resolution_results


def get_per_energy_metrics(results):
    per_energy_metrics = {}
    for pid, pid_results in results.items():
        all_pred = []
        all_true = []
        per_energy_metrics[pid] = {}
        for energy, energy_results in pid_results.items():
            res = {"pred": energy_results["pred"], "true": ak.sum(energy_results["true"] == 1, axis=-1)}
            all_pred.append(res["pred"])
            all_true.append(res["true"])
            per_energy_metrics[pid][energy] = collect_resolution_results(res)
        results_all = {
            "true": ak.concatenate(all_true, axis=-1),
            "pred": ak.concatenate(all_pred, axis=-1)
        }
        per_energy_metrics[pid]['global'] = collect_resolution_results(results_all)
    return per_energy_metrics
