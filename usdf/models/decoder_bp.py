import torch
import torch.nn as nn
import torch.nn.functional as F
from .se2_encoder import SE2Encoder

class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features,
                 activation="silu",
                 norm_style="layer",
                 norm_mode='post',
                 residual=False,
                 dropout=False,
                 dropout_prob=0.0,
                 last_layer=False,
                 weight_norm=False):
        super(MLPBlock, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.norm_mode = norm_mode
        self.norm_style = norm_style
        self.use_residual = residual and (in_features == out_features)
        self.weight_norm = weight_norm

        # initialize norm_layer to None; we'll create it only if weight_norm is False
        self.norm_layer = None

        linear = nn.Linear(in_features, out_features)
        if self.weight_norm and not last_layer:
            # apply weight normalization to the linear layer
            self.linear = nn.utils.weight_norm(linear)
            # when using weight_norm we will not create separate norm layers
            # self.norm_layer already None
        else:
            self.linear = linear
        self.activation = self.get_activation(activation)
        self.last_layer = last_layer

        # Create norm layer only if weight_norm is not used
        if not self.weight_norm:
            if norm_style == "batch":
                if norm_mode == 'pre':
                    self.norm_layer = nn.BatchNorm1d(in_features)
                else:
                    self.norm_layer = nn.BatchNorm1d(out_features)
            elif norm_style == "layer":
                if norm_mode == 'pre':
                    self.norm_layer = nn.LayerNorm(in_features)
                else:
                    self.norm_layer = nn.LayerNorm(out_features)
            else:
                raise ValueError(f"Unsupported norm style: {norm_style}")

        self.dropout = nn.Dropout(dropout_prob) if dropout == True else None

    def get_activation(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "silu":
            return F.silu
        elif activation == "tanh":
            return torch.tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        if self.last_layer:
            return self.linear(x)
        residual_in = x

        # Pre-norm pathway
        if self.norm_mode == 'pre':
            if self.norm_layer is not None:
                x = self.norm_layer(x)
            x = self.linear(x)
            x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout(x)
            if self.use_residual:
                x = x + residual_in
        elif self.norm_mode == 'post':
            x = self.linear(x)
            x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout(x)
            if self.use_residual:
                x = x + residual_in
            if self.norm_layer is not None:
                x = self.norm_layer(x)
        else:
            raise ValueError(f"Unsupported norm mode: {self.norm_mode}")
        return x

class Decoder(nn.Module):
    def __init__(self,
                 latent_size,
                 dims,
                 latent_in=[4],
                 activation="silu",
                 norm_style="layer",
                 norm_mode='post',
                 dropout=False,
                 dropout_prob=0.0,
                 residual=False,
                 weight_norm=False,
                 ):
        super(Decoder, self).__init__()

        self.latent_size = latent_size
        self.dims = dims
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.norm_style = norm_style
        self.norm_mode = norm_mode
        self.residual = residual
        self.activation = activation
        self.num_layers = len(dims)
        self.latent_in = latent_in
        self.weight_norm = weight_norm
        #self.encoder = SE2Encoder(latent_size=latent_size, hidden_dim=128, num_layers=3)

        layers = []
        dims = [latent_size + 3] + dims  + [1]
        for layer_idx in range(len(dims)-1):
            if layer_idx+1 in self.latent_in:
                out_dim = dims[layer_idx + 1] - dims[0]
            else:
                out_dim = dims[layer_idx + 1]
            in_dim = dims[layer_idx]
            layers.append(
                MLPBlock(
                    in_features=in_dim,
                    out_features=out_dim,
                    activation=activation,
                    norm_style=norm_style,
                    norm_mode=norm_mode,
                    residual=residual,
                    dropout=dropout,
                    dropout_prob=dropout_prob,
                    last_layer=(layer_idx == len(dims)-2),
                    weight_norm=weight_norm,
                )
            )
        self.network = nn.ModuleList(layers)

    def forward(self, x):
        #latent_vec = self.encoder(se2_vec)
        #x = torch.cat([latent_vec, x], dim=1)
        inputs = x
        for layer_idx, layer in enumerate(self.network):
            if layer_idx+1 in self.latent_in:
                x = layer(x)
                x = torch.cat([x, inputs], dim=-1)
            else:
                x = layer(x)
        return x
    
    def get_latent(self, se2_vec):
        return self.encoder(se2_vec)
