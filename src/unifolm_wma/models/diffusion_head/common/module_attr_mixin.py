import torch.nn as nn
import torch

class ModuleAttrMixin(nn.Module):

    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter(
            torch.zeros(1),
            requires_grad=False
        )

    @property
    def device(self):
        return self._dummy_variable.device

    @property
    def dtype(self):
        return self._dummy_variable.dtype
