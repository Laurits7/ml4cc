import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt
from ml4cc.tools.visualization.general import to_bh

hep.style.use(hep.styles.CMS)


class GlobalROCPlot:
    def __init__(self, figsize: tuple = (8, 8)):
        self.fig, self.ax = plt.subplots(figsize=figsize)

    def _add_curve(self, fpr: np.array, avg_tpr: np.array, std_tpr: np.array, label: str):
        self.ax.plot(fpr, avg_tpr, label=label)
        self.ax.fill_between(
            fpr,
            avg_tpr - std_tpr,
            avg_tpr + std_tpr,
            alpha=0.2,
            color=self.ax.lines[-1].get_color(),  # Use the same color as the last line
        )

    def plot_all_curves(self, results: dict, output_path: str = "") -> None:
        for pid, pid_result in results.items():
            mean_auc = pid_result['global']["AUC"]["mean"]
            std_auc = pid_result['global']["AUC"]["std"]
            label = f"{pid}: AUC={mean_auc:.3f} ± {std_auc:.3f}"
            self._add_curve(
                pid_result['global']["ROC"]["FPRs"],
                pid_result['global']["ROC"]["avg_TPRs"],
                pid_result['global']["ROC"]["std_TPRs"],
                label=label,
            )

        self.ax.set_xlabel("False Positive Rate (FPR)")
        self.ax.set_ylabel("True Positive Rate (TPR)")
        self.ax.legend()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        # self.ax.set_yscale("log")
        # self.ax.set_xscale("log")
        if output_path != "":
            plt.savefig(output_path, bbox_inches="tight")
            plt.close("all")
        else:
            plt.show()
            plt.close("all")


class MultiROCPlot:
    def __init__(self, pid: str = "muon", n_energies: int = 7, ncols: int = 3, figsize: tuple = (12, 12)):
        self.ncols = ncols
        self.pid = pid
        self.nrows = int(np.ceil(n_energies / ncols))
        self.fig, self.axis = plt.subplots(
            nrows=self.nrows, ncols=self.ncols, figsize=figsize, sharex=True, sharey=True
        )

    def _add_curve(
        self, fpr: np.array, tpr: np.array, auc: float, label: str, ax: plt.Axes, print_legend: bool = False
    ):
        ax.plot(fpr, tpr)
        ax.text(0.1, 0.8, f"AUC={auc:.3f}", fontsize=10)
        ax.set_title(label, fontsize=14)
        # ax.set_yscale("log")
        # ax.set_xscale("log")
        if print_legend:
            ax.legend(loc="upper right", fontsize=10)

    def plot_curves(self, results: dict, output_path: str = "") -> None:
        for idx, (energy, result) in enumerate(results.items()):
            ax = self.axis.flatten()[idx]
            label = f"{energy} GeV"
            self._add_curve(result["FPR"], result["TPR"], result["AUC"], label, ax, print_legend=idx == self.ncols)

        for ax in self.axis.flat:
            ax.label_outer()

        self.fig.text(0.04, 0.5, "True Positive Rate (TPR)", va="center", rotation="vertical", fontsize=20)
        self.fig.text(0.5, 0.02, "False Positive Rate (FPR)", ha="center", fontsize=20)
        if output_path != "":
            plt.savefig(output_path, bbox_inches="tight")
            plt.close("all")
        else:
            plt.show()
            plt.close("all")


class ClassifierScorePlot:
    def __init__(
        self,
        n_energies: int = 7,  # 7 for FCC, 6 for CEPC
        figsize: tuple = (9, 9),
        x_min: float = 0,
        x_max: float = 1,
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

    def _add_histogram(self, ax, preds: np.array, true: np.array, energy: str, print_legend: bool = False):
        sig_mask = true == 1
        bkg_mask = true == 0
        hep.histplot(
            to_bh(preds[sig_mask], bins=self.bins),
            ax=ax,
            density=True,
            histtype="fill",
            alpha=0.3,
            hatch="//",
            label="Signal",
        )
        hep.histplot(
            to_bh(preds[bkg_mask], bins=self.bins),
            ax=ax,
            density=True,
            histtype="fill",
            alpha=0.3,
            hatch="\\\\",
            label="Background",
        )
        ax.set_title(f"{energy} GeV", fontsize=16)
        ax.set_yscale('log')
        if print_legend:
            ax.legend(loc="upper right", fontsize=10)

    def plot_all_comparisons(self, results: dict, output_path: str = "") -> None:
        for idx, (energy, result) in enumerate(results.items()):
            if energy == "global": continue
            ax = self.axis.flatten()[idx]
            self._add_histogram(
                ax=ax, preds=result["pred"], true=result["true"], energy=energy, print_legend=idx == self.ncols
            )
        for ax in self.axis.flat:
            ax.label_outer()

        self.fig.text(0.04, 0.5, "Number of entries", va="center", rotation="vertical", fontsize=16)
        self.fig.text(0.5, 0.02, r"$\mathcal{D}_p$", ha="center", fontsize=16)

        if output_path != "":
            plt.savefig(output_path, bbox_inches="tight")
        else:
            plt.show()
        plt.close("all")


class AUCStackPlot:
    """For comparing the average AUC scores of different algorithms."""

    def __init__(
        self,
        color_mapping: dict = {},
        name_mapping: dict = {},
        marker_mapping: dict = {},
    ):
        self.color_mapping = color_mapping
        self.name_mapping = name_mapping
        self.marker_mapping = marker_mapping
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.pid_marker_mapping = {
            "muon": "^",
            "K": "s",
            "pi": "o"
        }

    def _add_line(self, results: dict, algorithm: str, y: int):
        for pid, pid_results in results.items():
            self.ax.errorbar(
                pid_results["AUC"]["mean"],
                y,
                xerr=pid_results["AUC"]["std"],
                label=self.name_mapping.get(algorithm, algorithm),
                color=self.color_mapping.get(algorithm, None),
                marker=self.pid_marker_mapping.get(pid, "o"),
                ls="",
                ms=10,
                capsize=5,
            )

    def plot_algorithms(self, results: dict, output_path: str = ""):
        yticklabels = []
        for idx, (algorithm, result) in enumerate(results.items()):
            yticklabels.append(self.name_mapping.get(algorithm, algorithm))
            self._add_line(result, algorithm=algorithm, y=idx)
        self.ax.set_xlabel(f"AUC score")
        self.ax.set_yticks(np.arange(len(yticklabels)))
        self.ax.set_yticklabels(yticklabels)
        if output_path != "":
            plt.savefig(output_path, bbox_inches="tight")
            plt.close("all")
        else:
            plt.show()
            plt.close("all")


class EnergyWiseAUC:
    """For comparing the average AUC scores of different energies of an algorithm."""

    def __init__(self, pids: list = None):
        self.pids = pids
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

    def plot_energies(self, results: dict, output_path: str = ""):
        for pid in self.pids:
            energies = np.array([key for key in results[pid].keys() if key != 'global'])
            pid_results = {key: results[pid][key] for key in energies}
            aucs = np.array([value["AUC"] for value in pid_results.values()])

            # Calculate mean AUC:
            mean_auc = np.mean(aucs)
            lower_than_average_mask = aucs < mean_auc
            for value, loc in zip(aucs[lower_than_average_mask], energies[lower_than_average_mask]):
                self.ax.vlines(loc, ymin=value, ymax=mean_auc, color='red', ls='--')
            for value, loc in zip(aucs[~lower_than_average_mask], energies[~lower_than_average_mask]):
                self.ax.vlines(loc, ymin=mean_auc, ymax=value, color='green', ls='--')
            self.ax.scatter(energies[lower_than_average_mask], aucs[lower_than_average_mask], color='red', s=80)
            self.ax.scatter(energies[~lower_than_average_mask], aucs[~lower_than_average_mask], color='green', s=80)
            self.ax.axhline(mean_auc, xmin=0, xmax=len(energies), ls='--', color='k')
        self.ax.set_ylabel(f"AUC score")
        self.ax.set_xlabel("Energy [GeV]")
        self.ax.set_xscale("log")
        self.fig.legend()
        if output_path != "":
            plt.savefig(output_path, bbox_inches="tight")
            plt.close("all")
        else:
            plt.show()
            plt.close("all")


class EffFakePlot:
    def __init__(self, eff_fake: str = "fake_rate"):
        """
        eff_fake: str
            Options:
                "fake_rate" for plotting fake rate vs energy,
                "efficiency" for plotting efficiency vs energy.
        """
        self.eff_fake = eff_fake
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

    def _add_line(self, result: dict, pid: str = ""):
        """Adds a line to the plot."""
        key = "FPRs" if self.eff_fake == "fake_rate" else "TPRs"
        self.ax.errorbar(result["energies"], result[key], xerr=0, ls="", marker="o", ms=5, capsize=5, label=pid)

    def plot_energies(self, results: dict, output_path: str = ""):
        for pid, pid_result in results.items():
            self._add_line(pid_result['global'], pid=pid)
        self.ax.set_xlabel("Energy [GeV]")
        ylabel = "Efficiency" if self.eff_fake == "efficiency" else "Fake rate"
        self.ax.set_ylabel(ylabel)
        self.ax.set_xscale('log')
        self.ax.legend()
        if output_path != "":
            plt.savefig(output_path, bbox_inches="tight")
            plt.close("all")
        else:
            plt.show()
            plt.close("all")
