import os
import json
import awkward as ak
from omegaconf import DictConfig
import ml4cc.tools.evaluation.general as g
import ml4cc.tools.evaluation.classification as c
import ml4cc.tools.evaluation.regression as r
from ml4cc.tools.visualization import losses as l
from ml4cc.tools.visualization import classification as vc
from ml4cc.tools.visualization import regression as vr


def prepare_regression_results(raw_results):
    res_results = {}
    for pid, pid_results in raw_results.items():
        res_results[pid] = {}
        for energy, energy_results in pid_results.items():
            print(energy_results.keys())
            res_results[pid][energy] = {
                'pred': ak.sum(energy_results['pred'] > 0.5, axis=-1),
                'true': energy_results['true']
            }
    return res_results


def evaluate_losses(
    metrics_path: str, model_name: str = "", loss_name: str = "BCE", results_dir: str = ""
):
    # Visualize losses for the training.
    losses = g.filter_losses(metrics_path=metrics_path)
    losses_output_path = os.path.join(results_dir, "losses.png")

    lp = l.LossesMultiPlot(loss_name=loss_name)
    loss_results = {model_name: {"val_loss": losses}}
    lp.plot_algorithms(results=loss_results, output_path=losses_output_path)


def evaluate_training(cfg: DictConfig, metrics_path: str):
    results_dir = os.path.join(cfg.training.results_dir, "two_step_minimal")
    os.makedirs(results_dir, exist_ok=True)
    evaluate_losses(
        metrics_path,
        model_name=cfg.models.two_step.peak_finding.model.name,
        loss_name="BCE",
        results_dir=results_dir,
    )

    # 1. Collect results
    prediction_dir = os.path.join(cfg.training.predictions_dir, "two_step_minimal")
    if not os.path.exists(prediction_dir):
        raise FileNotFoundError(f"Prediction directory {prediction_dir} does not exist.")
    raw_results = g.collect_all_results(predictions_dir=prediction_dir, cfg=cfg)

    # 2. Prepare results
    results = c.get_per_energy_metrics(results=raw_results, at_fakerate=0.01, at_efficiency=0.9, signal="both")

    results_json_path = os.path.join(results_dir, "results_pf.json")
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


    # Prepare raw results for regression
    res_results = prepare_regression_results(raw_results)
    r.get_per_energy_metrics(results)
    results_json_path = os.path.join(results_dir, "results_cl.json")
    with open(results_json_path, "wt") as out_file:
        json.dump(results, out_file)

    for pid in cfg.dataset.particle_types:
        pid_results = res_results[pid]
        multi_resolution_output_path = os.path.join(results_dir, f"{pid}_multi_resolution.png")
        mrp = vr.MultiResolutionPlot(n_energies=len(cfg.dataset.particle_energies), ncols=3)
        mrp.plot_all_resolutions(pid_results, output_path=multi_resolution_output_path)

    for pid in cfg.dataset.particle_types:
        pid_results = res_results[pid]
        multi_comparison_output_path = os.path.join(results_dir, f"{pid}_multi_comparison.png")
        mcp = vr.MultiComparisonPlot(n_energies=len(cfg.dataset.particle_energies), ncols=3)
        mcp.plot_all_comparisons(pid_results, output_path=multi_comparison_output_path)
