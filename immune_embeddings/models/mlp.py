from numbers import Real
from typing import Sequence, Union

from torch import nn

from immune_embeddings.utils import pairwise


class MLP(nn.Sequential):
    def __init__(self, *, in_dim: int, out_dim: int, layers: Sequence[int], width_factor: int = 1,
                 dropout: Union[float, Sequence[float]] = None, activation: nn.Module = None,
                 final_activation: nn.Module = None, flatten_input=False, squeeze_scalars=True):
        width_factor = max(int(width_factor), 1)
        all_layer_dims = [in_dim] + [l * width_factor for l in layers] + [out_dim]

        dropout = dropout or [0.0] * len(layers)
        if isinstance(dropout, Real):
            dropout = [float(dropout)] * len(layers)

        assert len(dropout) == len(layers)
        dropout.append(0.0)

        activation = activation or nn.ReLU()
        final_activation = final_activation or nn.Identity()
        activations = [activation] * len(layers) + [final_activation]

        all_layers = []
        if flatten_input:
            all_layers.append(nn.Flatten())
        for (i, (d_in, d_out)) in enumerate(pairwise(all_layer_dims)):
            all_layers.append(nn.Linear(d_in, d_out))
            if dropout[i] > 0.0:
                all_layers.append(nn.Dropout(dropout[i]))
            all_layers.append(activations[i])

        if squeeze_scalars and out_dim == 1:
            all_layers.append(nn.Flatten(-2))

        super(MLP, self).__init__(*all_layers)
