import os
import hydra
import numpy as np
import awkward as ak
from omegaconf import DictConfig
from ml4cc.tools.visualization import dataset as ds


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
    ds.visualize_num_peaks(full_info, pp_output_path, peak_type="primary", errorband=False)
    sp_output_path = os.path.join(cfg.evaluation.dataset.results_output_dir, "secondary_peaks.pdf")
    ds.visualize_num_peaks(full_info, sp_output_path, peak_type="secondary", errorband=False)
    for dataset_name, dataset_values in cfg.datasets.items():
        energies = dataset_values.particle_energies
        particle_types = dataset_values.particle_types
        for particle_type in particle_types:
            for energy in energies:
                os.makedirs(cfg.evaluation.dataset.results_output_dir, exist_ok=True)
                output_path = os.path.join(
                    cfg.evaluation.dataset.results_output_dir, f"{dataset_name}_{particle_type}_{energy}.png"
                )
                ds.visualize_primary_v_secondary_peaks_2d_histogram(
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
