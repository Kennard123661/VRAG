import torch
import torch.nn as nn
LEAKY_NEGATIVE_SLOPE = 1e-2


class Linear(nn.Module):
    def __init__(self, in_ndims: int, out_ndims: int, config: dict, bias: bool = True):
        super(Linear, self).__init__()
        layer = nn.Linear(in_features=in_ndims, out_features=out_ndims, bias=bias)
        net = [layer]

        if 'activation' in config:
            activation_layer = get_activation_layer(activation=config['activation'])
            net.append(activation_layer)
        self.net = nn.Sequential(*net)

    def forward(self, in_features: torch.Tensor):
        out_features = self.net(in_features)
        return out_features


class BatchLinear(nn.Module):
    def __init__(self, in_ndims: int, out_ndims: int, config: dict, bias: bool = True):
        super(BatchLinear, self).__init__()
        self.net = Linear(in_ndims=in_ndims, out_ndims=out_ndims, config=config, bias=bias)

    def forward(self, batch_features: list):
        batch_nfeatures = [features.shape[0] for features in batch_features]
        all_features = torch.cat(batch_features, dim=0)
        all_out_features = self.net(all_features)

        batch_out_features = []
        start = 0
        for nfeatures in batch_nfeatures:
            end = nfeatures + start
            out_features = all_out_features[start:end]
            start = end
            batch_out_features.append(out_features)
        return batch_out_features


def get_activation_layer(activation: str):
    if activation == 'elu':
        return nn.ELU(inplace=True)
    else:
        raise NotImplementedError
