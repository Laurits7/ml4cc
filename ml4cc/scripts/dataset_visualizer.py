import os
import hydra
import awkward as ak
import numpy as np
import glob
import matplotlib.pyplot as plt
import mplhep as hep
from omegaconf import DictConfig

hep.style.use(hep.styles.CMS)


def analyze_cluster_counts(cluster_counts: ak.Array) -> dict:
    info = {"mean": np.mean(cluster_counts), "stdev": np.std(cluster_counts)}
    return info


def analyze_FCC_case_data(cfg: DictConfig, particle: str, energy: str) -> dict:
    path = os.path.join(cfg.data_dir, "one_step", "test", f"{energy}_1.parquet")
    data = ak.from_parquet(path)
    num_primary = ak.sum(data.target == 1, axis=-1)
    num_secondary = ak.sum(data.target == 2, axis=-1)
    secondary_peak_info = analyze_cluster_counts(num_secondary)
    primary_peak_info = analyze_cluster_counts(num_primary)
    info = {
        "raw_num_primary": num_primary,
        "raw_num_secondary": num_secondary,
        "primary_peak_info": primary_peak_info,
        "secondary_peak_info": secondary_peak_info,
    }
    return info


def analyze_CEPC_case_data(cfg: DictConfig, particle: str, energy: str) -> dict:
    path = os.path.join(cfg.data_dir, "one_step", "test", f"signal_{particle}_{energy}_0.parquet")
    data = ak.from_parquet(path)
    num_primary = ak.sum(data.target == 1, axis=-1)
    num_secondary = ak.sum(data.target == 2, axis=-1)
    secondary_peak_info = analyze_cluster_counts(num_secondary)
    primary_peak_info = analyze_cluster_counts(num_primary)
    info = {
        "raw_num_primary": num_primary,
        "raw_num_secondary": num_secondary,
        "primary_peak_info": primary_peak_info,
        "secondary_peak_info": secondary_peak_info,
    }
    return info


def collect_energy_wise_info(particle_type_info, peak_type):
    plotting_info = {"energy": [], "mean_values": [], "min_values": [], "max_values": [], "stdev": []}
    for energy, energy_info in particle_type_info.items():
        if energy_info == {}:
            continue
        plotting_info["energy"].append(energy)
        plotting_info["mean_values"].append(energy_info[f"{peak_type}_peak_info"]["mean"])
        plotting_info["min_values"].append(
            energy_info[f"{peak_type}_peak_info"]["mean"] - energy_info[f"{peak_type}_peak_info"]["stdev"]
        )
        plotting_info["max_values"].append(
            energy_info[f"{peak_type}_peak_info"]["mean"] + energy_info[f"{peak_type}_peak_info"]["stdev"]
        )
        plotting_info["stdev"].append(energy_info[f"{peak_type}_peak_info"]["stdev"])
    return plotting_info


def visualize_num_peaks(full_info, output_path: str, peak_type="primary", errorband=True):
    fig, ax = plt.subplots(figsize=(10, 10))
    p_name_mapping = {"muon": r"$\mu^{\pm}$", "K": r"$K^{\pm}$", "pi": r"$\pi^{\pm}$"}
    marker_mapping = {"CEPC": "v", "FCC": "^"}
    color_map = {"K": "g", "pi": "r", "muon": "b"}
    for experiment_name, experiment_info in full_info.items():
        for particle_type, particle_type_info in experiment_info.items():
            plotting_info = collect_energy_wise_info(particle_type_info, peak_type)
            plt.plot(
                plotting_info["energy"],
                plotting_info["mean_values"],
                ls="-",
                marker=marker_mapping[experiment_name],
                label=f"{experiment_name}: {p_name_mapping[particle_type]}",
                color=color_map[particle_type],
            )
            if errorband:
                ax.fill_between(
                    x=plotting_info["energy"],
                    y1=plotting_info["min_values"],
                    y2=plotting_info["max_values"],
                    color=color_map[particle_type],
                    alpha=0.3,
                )
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    fig.savefig(output_path)


def visualize_primary_v_secondary_peaks_2d_histogram(
    full_info: dict, experiment: str, particle_type: str, energy: float, output_path: str, max_peaks: int = 28
):
    x = np.array(full_info[experiment][particle_type][energy]["raw_num_primary"])
    y = np.array(full_info[experiment][particle_type][energy]["raw_num_secondary"])

    fig = plt.figure(figsize=(16, 16))
    grid = plt.GridSpec(4, 4, hspace=0.0, wspace=0.0)

    main_ax = fig.add_subplot(grid[1:, :-1])
    y_hist = fig.add_subplot(grid[1:, -1], sharey=main_ax)
    x_hist = fig.add_subplot(grid[0, :-1], sharex=main_ax)

    xbins = ybins = np.linspace(0, max_peaks + 1, max_peaks)
    H, x_edges, y_edges = np.histogram2d(x, y, bins=xbins, density=None, weights=None)
    hep.hist2dplot(H, x_edges, y_edges, ax=main_ax, cbar=False, cmap="Greys")

    main_ax.set_xlabel("Number of Primary Peaks")
    main_ax.set_ylabel("Number of Secondary Peaks")

    x_hist.hist(x, bins=xbins, color="gray", histtype="step")
    x_hist.tick_params(top=True, labeltop=False, bottom=True, labelbottom=False)
    x_hist.tick_params(left=True, labelleft=False, right=True, labelright=True)

    y_hist.hist(y, bins=ybins, orientation="horizontal", color="gray", histtype="step")
    y_hist.tick_params(top=True, labeltop=True, bottom=True, labelbottom=False)
    y_hist.tick_params(left=True, labelleft=False, right=True, labelright=False)
    plt.savefig(output_path)


def accumulate_statistics(cfg: DictConfig) -> dict:
    full_info = {}
    for dataset_name, dataset_values in cfg.datasets.items():
        full_info[dataset_name] = {}
        energies = dataset_values.particle_energies
        particle_types = dataset_values.particle_types
        for particle_type in particle_types:
            full_info[dataset_name][particle_type] = {}
            for energy in energies:
                if dataset_name == "CEPC":
                    info = analyze_CEPC_case_data(cfg, particle_type, energy)
                elif dataset_name == "FCC":
                    info = analyze_FCC_case_data(cfg, particle_type, energy)
                else:
                    raise ValueError("Something went wrong, experiment config not found")
                full_info[dataset_name][particle_type][energy] = info
    return full_info


def visualize_all(full_info: dict, cfg: DictConfig) -> None:
    pp_output_path = os.path.join(cfg.evaluation.dataset.results_output_dir, "primary_peaks.pdf")
    visualize_num_peaks(full_info, pp_output_path, peak_type="primary", errorband=False)
    sp_output_path = os.path.join(cfg.evaluation.dataset.results_output_dir, "secondary_peaks.pdf")
    visualize_num_peaks(full_info, sp_output_path, peak_type="secondary", errorband=False)
    for dataset_name, dataset_values in cfg.datasets.items():
        energies = dataset_values.particle_energies
        particle_types = dataset_values.particle_types
        for particle_type in particle_types:
            for energy in energies:
                os.makedirs(cfg.evaluation.dataset.results_output_dir, exist_ok=True)
                output_path = os.path.join(
                    cfg.evaluation.dataset.results_output_dir, f"{dataset_name}_{particle_type}_{energy}.png"
                )
                visualize_primary_v_secondary_peaks_2d_histogram(
                    full_info=full_info,
                    experiment=dataset_name,
                    particle_type=particle_type,
                    energy=energy,
                    output_path=output_path,
                    max_peaks=45,
                )


@hydra.main(config_path="../config", config_name="main.yaml", version_base=None)
def main(cfg: DictConfig):
    full_info = accumulate_statistics(cfg=cfg)
    visualize_all(full_info=full_info, cfg=cfg)


if __name__ == "__main__":
    main()  # pylint: disable=E1120
