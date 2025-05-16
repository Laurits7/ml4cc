import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt

hep.style.use(hep.styles.CMS)


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
