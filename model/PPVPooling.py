import torch
from torch.nn import Module


class PPVPooling(Module):
    """
    Postive Porpotion Value Pooling Layer

    - given the bias (defaults 0) and channel format, the layer applies change on the input tensor's dims.

    - for channels_first=True, we compress the 2th, 3th dims
      else, we compress the 1th and 2th dims.
    """
    def __init__(self, bias=0, channels_first=True, **kwargs):
        super(PPVPooling, self).__init__()
        self.bias = bias
        self.dims = [2, 3] if channels_first else [1, 2]

    def forward(self, x):
        handeledx = torch.greater(x, self.bias).float()
        ppvoutput = torch.mean(handeledx, dim=self.dims)
        return ppvoutput
