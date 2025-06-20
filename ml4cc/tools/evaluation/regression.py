import numpy as np


def calculate_resolution(truth: np.array, preds: np.array) -> np.array:
    """We calculate the resolution as IQR/median as stdev is more affected by outliers"""
    ratios = truth / preds
    iqr = np.quantile(ratios, 0.75) - np.quantile(ratios, 0.25)
    median = np.quantile(ratios, 0.5)
    resolution = iqr / median
    return resolution, ratios


def collect_resolution_results(results: dict) -> dict:  # TODO: Unfinished
    """Collects the resolution results from different energies"""
    resolution_results = {}
    for energy, result in results.items():
        resolution, ratios = calculate_resolution(result["truth"], result["preds"])
        resolution_results[energy] = {
            "resolution": resolution,
            "ratios": ratios,
        }
    return resolution_results


def get_per_energy_metrics(results):
    print(results)
