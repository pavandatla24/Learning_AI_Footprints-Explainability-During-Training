from typing import Dict, List
import torch

class FootprintCollector:
    """
    Collects activations (and later gradients / batch summaries) during training.

    For Week 1 this is just a skeleton; real logic comes in Week 3.
    """
    def __init__(self):
        self.raw_activations: List[torch.Tensor] = []

    def hook(self, module, module_in, module_out):
        """
        Hook function to attach to a model layer.
        PyTorch will call this every time the layer runs.
        """
        # Detach to avoid keeping computation graph
        self.raw_activations.append(module_out.detach().cpu())

    def reset(self):
        self.raw_activations = []

    def get_collected(self) -> List[torch.Tensor]:
        return self.raw_activations
