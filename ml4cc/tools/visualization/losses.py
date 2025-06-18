import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt

hep.style.use(hep.styles.CMS)


class LossesMultiPlot:
    def __init__(self,
                 plot_train_losses: bool = False,
                 loss_name: str = "MSE",
                 color_mapping: dict = {},
                 name_mapping: dict = {},
                 x_max: int = -1
                 ):
        self.plot_train_losses = plot_train_losses
        self.loss_name = loss_name
        self.color_mapping = color_mapping
        self.name_mapping = name_mapping
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.x_max = x_max

    def _add_line(self, results: dict, algorithm: str):
        """Adds a line to the plot."""
        if self.plot_train_losses:
            self.ax.plot(
                results['train_loss'],
                ls="--",
                color=self.color_mapping.get(algorithm, None)
            ) # Train loss always with dashed line
        self.ax.plot(
            results['val_loss'],
            label=self.name_mapping.get(algorithm, algorithm),
            ls="-",
            color=self.color_mapping.get(algorithm, None)
        ) # Val loss always with solid line
        self.ax.legend()

    def plot_algorithms(self, results: dict, output_path: str = ""):
        for idx, (algorithm, result) in enumerate(results.items()):
            self._add_line(result, algorithm=algorithm)
        self.ax.set_yscale("log")
        self.ax.set_ylabel(f"{self.loss_name} loss [a.u.]")
        self.ax.set_xlabel("epoch")
        self.ax.set_xlim(0, self.x_max if self.x_max > 0 else 100)
        if output_path != "":
            plt.savefig(output_path, bbox_inches="tight")
            plt.close("all")
        else:
            plt.show()
            plt.close("all")


class LossesStackPlot:
    def __init__(self,
                 loss_name: str = "MSE",
                 color_mapping: dict = {},
                 name_mapping: dict = {},
                 marker_mapping: dict = {},
                 ):
        self.loss_name = loss_name
        self.color_mapping = color_mapping
        self.name_mapping = name_mapping
        self.marker_mapping = marker_mapping
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

    def _add_line(self, results: dict, algorithm: str, y: int):
        self.ax.errorbar(
            np.mean(results["best_losses"]),
            y,
            xerr=np.std(results["best_losses"]),
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
        self.ax.set_xlabel(f"{self.loss_name} loss [a.u.]")
        self.ax.set_yticks(np.arange(len(yticklabels)))
        self.ax.set_yticklabels(yticklabels)
        if output_path != "":
            plt.savefig(output_path, bbox_inches="tight")
            plt.close("all")
        else:
            plt.show()
            plt.close("all")
