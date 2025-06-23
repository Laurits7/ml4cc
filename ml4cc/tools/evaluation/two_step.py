import os
import json
from omegaconf import DictConfig
import ml4cc.tools.evaluation.general as g
import ml4cc.tools.evaluation.classification as c
import ml4cc.tools.evaluation.regression as r
from ml4cc.tools.visualization import classification as vc
from ml4cc.tools.visualization import regression as vr


def evaluate_training(cfg: DictConfig, metrics_path: str, stage: str):
    if stage == "peak_finding":
        results_dir = os.path.join(cfg.training.results_dir, "two_step_pf")
        evaluate_peak_finding(cfg, metrics_path, results_dir=results_dir)
    elif stage == "classification":
        results_dir = os.path.join(cfg.training.results_dir, "two_step_cl")
        evaluate_clusterization(cfg, metrics_path, results_dir=results_dir)
    else:
        raise ValueError(f"Incorrect evaluation stage: {stage}")


def evaluate_peak_finding(cfg: DictConfig, metrics_path: str, results_dir: str):
    # 0. Visualize losses for the training.
    os.makedirs(results_dir, exist_ok=True)

    g.evaluate_losses(
        metrics_path,
        model_name=cfg.models.two_step.peak_finding.model.name,
        loss_name="BCE",
        results_dir=results_dir,
    )

    # 1. Collect results
    prediction_dir = os.path.join(cfg.training.predictions_dir, "two_step_pf")
    if not os.path.exists(prediction_dir):
        raise FileNotFoundError(f"Prediction directory {prediction_dir} does not exist.")
    raw_results = g.collect_all_results(predictions_dir=prediction_dir, cfg=cfg)

    # 2. Prepare results
    results = c.get_per_energy_metrics(results=raw_results, at_fakerate=0.01, at_efficiency=0.9, signal="both")

    results_json_path = os.path.join(results_dir, "results.json")
    with open(results_json_path, "wt") as out_file:
        json.dump(results, out_file)

    # 3. Visualize results
    for pid in cfg.dataset.particle_types:
        pid_results = results[pid]
        csp_output_path = os.path.join(results_dir, "classifier_scores.png")
        csp = vc.ClassifierScorePlot(n_energies=len(cfg.dataset.particle_energies))
        csp.plot_all_comparisons(results=pid_results, output_path=csp_output_path)

    asp_output_path = os.path.join(results_dir, "AUC_stack.png")
    ewa = vc.EnergyWiseAUC(pids=cfg.dataset.particle_types)
    ewa.plot_energies(results, output_path=asp_output_path)

    fr_output_path = os.path.join(results_dir, "fake_rate.png")
    frp = vc.EffFakePlot(eff_fake="fake_rate")
    frp.plot_energies(results["global"], output_path=fr_output_path)

    eff_output_path = os.path.join(results_dir, "efficiency.png")
    efp = vc.EffFakePlot(eff_fake="efficiency")
    efp.plot_energies(results["global"], output_path=eff_output_path)

    for pid in cfg.dataset.particle_types:
        pid_results = results[pid]
        multiroc_output_path = os.path.join(results_dir, f"{pid}_multi_roc.png")
        mroc = vc.MultiROCPlot(pid=pid, n_energies=len(cfg.dataset.particle_energies), ncols=3)
        mroc.plot_curves(pid_results, output_path=multiroc_output_path)

    global_roc_output_path = os.path.join(results_dir, "global_roc.png")
    grp = vc.GlobalROCPlot()
    grp.plot_all_curves(results["global"], output_path=global_roc_output_path)


def evaluate_clusterization(cfg: DictConfig, metrics_path: str, results_dir: str):
    # Visualize losses for the training.
    g.evaluate_losses(metrics_path, model_name=cfg.models.clusterization.model_name, loss_name="MSE")

    # 1. Collect results
    prediction_dir = os.path.join(cfg.training.predictions_dir, "two_step_cl")
    if not os.path.exists(prediction_dir):
        raise FileNotFoundError(f"Prediction directory {prediction_dir} does not exist.")
    raw_results = g.collect_all_results(predictions_dir=prediction_dir, cfg=cfg)

    # Evaluate model performance.
    # 2. Prepare results
    results = r.get_per_energy_metrics(results=raw_results)

    results_json_path = os.path.join(results_dir, "results.json")
    with open(results_json_path, "wt") as out_file:
        json.dump(results, out_file)


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
