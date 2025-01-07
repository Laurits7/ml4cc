import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt
hep.style.use(hep.styles.CMS)


def plot_loss_evolution(
        val_loss: np.array,
        train_loss: np.array,
        output_path: str = "",
        loss_name: str = "MSE"
):
    """ Plots the evolution of train and validation loss.

    Parameters:
        val_loss : np.array
            Validation losses for the epochs
        train_loss : np.array
            Training losses for the epochs
        output_path : str
            [default: ''] Path where figure will be saved. If empty string, figure will not be saved
        loss_name : str
            [default: "MSE"] Loss function name used for the training

    Returns:
        None
    """
    # if multirun case?
    plt.plot(val_loss, label="val_loss", color='k')
    if train_loss is not None:
        plt.plot(train_loss, label="train_loss", ls="--", color='k')
    plt.grid()
    plt.yscale('log')
    plt.ylabel(f'{loss_name} loss [a.u.]')
    plt.xlabel('epoch')
    plt.xlim(0, len(val_loss))
    plt.legend()
    plt.savefig(output_path)
    if output_path != '':
        plt.savefig(output_path, bbox_inches='tight')
    plt.close("all")
