import os
import json
from ml4cc.tools.evaluation import regression as r
from ml4cc.tools.evaluation import general as g
from ml4cc.tools.visualization import regression as vr
from ml4cc.tools.evaluation.general import NumpyEncoder


def evaluate_training(cfg, metrics_path):
    results_dir = cfg.training.results_dir
    os.makedirs(results_dir, exist_ok=True)
    predictions_dir = cfg.training.predictions_dir

    g.evaluate_losses(metrics_path, model_name=cfg.models.one_step.model.name, loss_name="MSE")

    # 1. Collect results
    if not os.path.exists(predictions_dir):
        raise FileNotFoundError(f"Prediction directory {predictions_dir} does not exist.")
    raw_results = g.collect_all_results(predictions_dir=predictions_dir, cfg=cfg)

    # Evaluate model performance.
    # 2. Prepare results
    results = r.get_per_energy_metrics(results=raw_results)

    results_json_path = os.path.join(results_dir, "results.json")
    with open(results_json_path, "wt") as out_file:
        json.dump(
            results,
            out_file,
            indent=4,
            cls=NumpyEncoder
        )


    for pid in cfg.dataset.particle_types:
        pid_results = results[pid]
        multi_resolution_output_path = os.path.join(results_dir, f"{pid}_multi_resolution.png")
        mrp = vr.MultiResolutionPlot(n_energies=len(cfg.dataset.particle_energies), ncols=3)
        mrp.plot_all_resolutions(pid_results, output_path=multi_resolution_output_path)

    for pid in cfg.dataset.particle_types:
        pid_results = results[pid]
        multi_comparison_output_path = os.path.join(results_dir, f"{pid}_multi_comparison.png")
        mcp = vr.MultiComparisonPlot(n_energies=len(cfg.dataset.particle_energies), ncols=3)
        mcp.plot_all_comparisons(pid_results, output_path=multi_comparison_output_path)
