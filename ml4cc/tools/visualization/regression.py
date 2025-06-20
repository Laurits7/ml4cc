import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt
from ml4cc.tools.visualization.general import to_bh

hep.style.use(hep.styles.CMS)


class MultiResolutionPlot:
    def __init__(
        self,
        n_energies: int = 7,  # 7 for FCC, 6 for CEPC
        figsize: tuple = (9, 9),
        x_min: float = 0.5,
        x_max: float = 1.5,
        num_bins: int = 15,
        ncols: int = 3,
    ):
        self.x_min = x_min
        self.ncols = ncols
        self.nrows = int(np.ceil(n_energies / ncols))
        self.x_max = x_max
        self.num_bins = num_bins
        self.fig, self.axis = plt.subplots(
            nrows=self.nrows, ncols=self.ncols, figsize=figsize, sharex=True, sharey=True
        )

    def _add_histogram(self, ax, ratios: np.array, resolution: float, energy: str):
        bins = np.linspace(self.x_min, self.x_max, self.num_bins)
        hep.histplot(to_bh(ratios, bins=bins), ax=ax, density=True)
        ax.axvline(x=1, ls="--", color="k")
        ax.set_title(f"{energy} GeV")
        ax.text(0.95, 0.95, f"IQR={resolution:.2f}", ha="right", va="top", transform=ax.transAxes, fontsize=10)

    def plot_all_resolutions(self, results: dict, output_path: str = "") -> None:
        for idx, (energy, result) in enumerate(results.items()):
            ax = self.axis.flatten()[idx]
            self._add_histogram(ax=ax, ratios=result["ratios"], resolution=result["resolution"], energy=energy)
        for ax in self.axis.flat:
            ax.label_outer()

        self.fig.text(0.04, 0.5, "Number of entries", va="center", rotation="vertical", fontsize=12)
        self.fig.text(0.5, 0.02, r"$n_{cls}^{true}/n_{cls}^{pred}$", ha="center", fontsize=12)

        if output_path != "":
            plt.savefig(output_path, bbox_inches="tight")
        else:
            plt.show()
        plt.close("all")


class MultiComparisonPlot:
    def __init__(
        self,
        n_energies: int = 7,  # 7 for FCC, 6 for CEPC
        figsize: tuple = (9, 9),
        x_min: float = 0,
        x_max: float = 50,
        num_bins: int = 25,
        ncols: int = 3,
    ):
        self.x_min = x_min
        self.ncols = ncols
        self.nrows = int(np.ceil(n_energies / ncols))
        self.x_max = x_max
        self.num_bins = num_bins
        self.fig, self.axis = plt.subplots(
            nrows=self.nrows, ncols=self.ncols, figsize=figsize, sharex=True, sharey=True
        )
        self.bins = np.linspace(self.x_min, self.x_max, self.num_bins)

    def _add_histograms(self, ax, true: np.array, pred: float, energy: str, print_legend: bool = False):
        hep.histplot(
            to_bh(true, bins=self.bins), ax=ax, density=True, histtype="fill", alpha=0.3, hatch="//", label="Target"
        )
        hep.histplot(
            to_bh(pred, bins=self.bins),
            ax=ax,
            density=True,
            histtype="fill",
            alpha=0.3,
            hatch="\\\\",
            label="Reconstructed",
        )
        ax.set_title(f"{energy} GeV")
        if print_legend:
            ax.legend(loc="upper right", fontsize=10)

    def plot_all_comparisons(self, results: dict, output_path: str = "") -> None:
        for idx, (energy, result) in enumerate(results.items()):
            ax = self.axis.flatten()[idx]
            self._add_histograms(
                ax=ax, true=result["true"], pred=result["pred"], energy=energy, print_legend=idx == self.ncols
            )
        for ax in self.axis.flat:
            ax.label_outer()

        self.fig.text(0.04, 0.5, "Number of entries", va="center", rotation="vertical", fontsize=12)
        self.fig.text(0.5, 0.02, "Number of primary clusters", ha="center", fontsize=12)

        if output_path != "":
            plt.savefig(output_path, bbox_inches="tight")
        else:
            plt.show()
        plt.close("all")


class RegressionStackPlot:
    # TODO: This is only for a single energy, should maybe also do for multiple energies?
    def __init__(
        self,
        normalize_by_median: bool = True,
        color_mapping: dict = {},
        name_mapping: dict = {},
        marker_mapping: dict = {},
    ):
        self.normalize_by_median = normalize_by_median
        self.color_mapping = color_mapping
        self.name_mapping = name_mapping
        self.marker_mapping = marker_mapping
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

    def _add_line(self, results: dict, algorithm: str, y: int):
        self.ax.errorbar(
            results["median"],
            y,
            xerr=results["IQR"],
            label=self.name_mapping.get(algorithm, algorithm),
            color=self.color_mapping.get(algorithm, None),
            marker=self.marker_mapping.get(algorithm, "o"),
            ls="",
            ms=10,
            capsize=5,
        )

    def plot_algorithms(self, results: dict, output_path: str = ""):
        yticklabels = []
        for idx, (algorithm, result) in enumerate(results.items()):
            yticklabels.append(algorithm)
            self._add_line(result, algorithm=algorithm, y=idx)
        self.ax.axvline(1, color="k", ls="--")
        self.ax.set_xlabel("Response")
        self.ax.set_yticks(np.arange(len(yticklabels)))
        self.ax.set_yticklabels(yticklabels)
        if output_path != "":
            plt.savefig(output_path, bbox_inches="tight")
            plt.close("all")
        else:
            plt.show()
            plt.close("all")
