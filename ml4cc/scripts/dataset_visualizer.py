import os
import hydra
import awkward as ak
import numpy as np
# from typing import Tuple
from omegaconf import DictConfig


def analyze_cluster_counts(cluster_counts: ak.Array) -> dict:
    info = {
        "mean": np.mean(cluster_counts),
        "stdev": np.std(cluster_counts)
    }
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
        "secondary_peak_info": secondary_peak_info
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
        "secondary_peak_info": secondary_peak_info
    }
    return info


def compile_num_cluster_v_energy_info():
    pass


@hydra.main(config_path="../config", config_name="main.yaml", version_base=None)
def main(cfg: DictConfig):
    full_info = {}
    for dataset_name, dataset_values in cfg.datasets.items():
        full_info[dataset_name] = {}
        energies = dataset_values.particle_energies
        particle_types = dataset_values.particle_types
        for particle_type in particle_types:
            full_info[dataset_name][particle_type] = {}
            for energy in energies:
                if dataset_name == 'CEPC':
                    info = analyze_CEPC_case_data(cfg, particle_type, energy)
                elif dataset_name == "FCC":
                    info = analyze_FCC_case_data(cfg, particle_type, energy)
                full_info[dataset_name][particle_type][energy] = info
    




    # TODO: Load X number of CEPC files: 
    #       - Plot number of primary clusters mean+std for each Particle type and energy






if __name__ == "__main__":
    main()  # pylint: disable=E1120