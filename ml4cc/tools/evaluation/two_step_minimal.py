import os
import json
import awkward as ak
from omegaconf import DictConfig
import ml4cc.tools.evaluation.general as g
import ml4cc.tools.evaluation.classification as c
import ml4cc.tools.evaluation.regression as r
from ml4cc.tools.visualization import classification as vc
from ml4cc.tools.visualization import regression as vr
from ml4cc.tools.evaluation.general import NumpyEncoder


def prepare_regression_results(raw_results):
    res_results = {}
    for pid, pid_results in raw_results.items():
        res_results[pid] = {}
        all_true = []
        all_pred = []
        for energy, energy_results in pid_results.items():
            pred = ak.sum(energy_results['pred'] > 0.5, axis=-1)
            true = ak.sum(energy_results['true'] == 1, axis=-1)
            resolution, median, ratios = r.calculate_resolution(true, pred)
            res_results[pid][energy] = {
                'pred': pred,
                'true': true,
                'ratios': ratios,
                'resolution': resolution,
                'median': median
            }
            all_true.append(true)
            all_pred.append(pred)
        results_all = {
            "true": ak.concatenate(all_true, axis=-1),
            "pred": ak.concatenate(all_pred, axis=-1)
        }
        res_results[pid]['global'] = r.collect_resolution_results(results_all)
    return res_results


def evaluate_training(cfg: DictConfig, metrics_path: str):
    results_dir = os.path.join(cfg.training.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    g.evaluate_losses(
        metrics_path,
        model_name=cfg.models.two_step.peak_finding.model.name,
        loss_name="BCE",
        results_dir=results_dir,
    )

    # 1. Collect results
    prediction_dir = os.path.join(cfg.training.predictions_dir)
    if not os.path.exists(prediction_dir):
        raise FileNotFoundError(f"Prediction directory {prediction_dir} does not exist.")
    raw_results = g.collect_all_results(predictions_dir=prediction_dir, cfg=cfg)

    # 2. Prepare results
    cls_results = c.get_per_energy_metrics(results=raw_results, at_fakerate=0.01, at_efficiency=0.9, signal="primary")

    results_json_path = os.path.join(results_dir, "results_pf.json")
    with open(results_json_path, "wt") as out_file:
        json.dump(
            cls_results,
            out_file,
            indent=4,
            cls=NumpyEncoder
        )

    # 3. Visualize results
    for pid in cfg.dataset.particle_types:
        pid_results = cls_results[pid]
        csp_output_path = os.path.join(results_dir, "classifier_scores.png")
        csp = vc.ClassifierScorePlot(n_energies=len(cfg.dataset.particle_energies))
        csp.plot_all_comparisons(results=pid_results, output_path=csp_output_path)

    asp_output_path = os.path.join(results_dir, "AUC_stack.png")
    ewa = vc.EnergyWiseAUC(pids=cfg.dataset.particle_types)
    ewa.plot_energies(cls_results, output_path=asp_output_path)

    fr_output_path = os.path.join(results_dir, "fake_rate.png")
    frp = vc.EffFakePlot(eff_fake="fake_rate")
    frp.plot_energies(cls_results, output_path=fr_output_path)

    eff_output_path = os.path.join(results_dir, "efficiency.png")
    efp = vc.EffFakePlot(eff_fake="efficiency")
    efp.plot_energies(cls_results, output_path=eff_output_path)

    for pid in cfg.dataset.particle_types:
        energies = [key for key in cls_results[pid].keys() if key != 'global']
        pid_results = {key: cls_results[pid][key] for key in energies}
        multiroc_output_path = os.path.join(results_dir, f"{pid}_multi_roc.png")
        mroc = vc.MultiROCPlot(pid=pid, n_energies=len(cfg.dataset.particle_energies), ncols=3)
        mroc.plot_curves(pid_results, output_path=multiroc_output_path)

    global_roc_output_path = os.path.join(results_dir, "global_roc.png")
    grp = vc.GlobalROCPlot()
    grp.plot_all_curves(cls_results, output_path=global_roc_output_path)


    # Prepare raw results for regression
    res_results = prepare_regression_results(raw_results)
    results_json_path = os.path.join(results_dir, "results_cl.json")
    with open(results_json_path, "wt") as out_file:
        json.dump(
            res_results,
            out_file,
            indent=4,
            cls=NumpyEncoder
        )

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

    # Merge the two results to one results.json
    all_results = {}
    for pid in cfg.dataset.particle_types:
        all_results[pid] = {
            "global": {
                "resolution": res_results[pid]["global"]["resolution"],
                "median": res_results[pid]["global"]["median"],
                "AUC": cls_results[pid]["global"]["AUC"]
            }
        }
    all_results_json_path = os.path.join(results_dir, "results.json")
    with open(all_results_json_path, "wt") as out_file:
        json.dump(
            all_results,
            out_file,
            indent=4,
            cls=NumpyEncoder
        )