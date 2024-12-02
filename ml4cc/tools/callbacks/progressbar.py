import sys
from lightning.pytorch.callbacks import ProgressBar


class LitProgressBar(ProgressBar):

    def __init__(self):
        super().__init__()
        self.enable = True

    def disable(self):
        self.enable = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        percent = (batch_idx / self.total_train_batches) * 100
        sys.stdout.flush()
        sys.stdout.write(f"{percent:.01f} percent complete \r")
