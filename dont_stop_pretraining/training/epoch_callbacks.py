from allennlp.training.trainer import EpochCallback
from typing import Dict, Any
import os
import torch

@EpochCallback.register("save_supermasks")
class SaveSupermasks(EpochCallback):
    """
    An optional callback that you can pass to the `GradientDescentTrainer` that will be called at
    the end of every epoch (and before the start of training, with `epoch=-1`). We have no default
    implementation of this, but you can implement your own callback and do whatever you want, such
    as additional modifications of the trainer's state in between epochs.
    """

    def __call__(
        self, trainer: "GradientDescentTrainer", metrics: Dict[str, Any], epoch: int
    ) -> None:
        supermasks = trainer.model.get_all_supermasks()
        outdir = os.path.join(trainer._serialization_dir, "supermasks", f"epoch_{epoch}")
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        for name, curr_mask, _, _ in supermasks:
            torch.save(curr_mask, os.path.join(outdir, name))
