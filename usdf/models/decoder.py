import torch
import torch.nn as nn
import torch.nn.functional as F
from .se2_encoder import SE2Encoder
import numpy as np
from collections import OrderedDict

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
                print("No norm layer")
                self.norm_layer = None

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

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 is_first=False,
                 is_last=False,
                 omega_0=30,
                 norm_style="layer",
                 residual=False,
                 norm_mode='post',
                 dropout=False,
                 dropout_prob=0.0,
                 weight_norm=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.norm_mode = norm_mode
        self.norm_style = norm_style
        self.weight_norm = weight_norm
        self.last_layer = is_last
        self.use_residual = residual and (in_features == out_features)

        self.dropout = nn.Dropout(dropout_prob) if dropout == True else None

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                print("Initializing first layer")
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                print("Initializing hidden layer")
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, inputs):
        x = inputs
        if self.last_layer:
            return self.linear(x)
        return torch.sin(self.omega_0 * self.linear(x))

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
                 mode ='mlp',
                 first_omega_0=30,
                 hidden_omega_0=30
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
        self.encoder = SE2Encoder(mode=mode, latent_size=latent_size, hidden_dim=128, num_layers=3)
        if mode == 'mlp':
            print("Using MLP Decoder")
        else:
            print("Using Sine Decoder")
        self.mode = mode
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0

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
                ) if mode == 'mlp' else
                SineLayer(
                    in_features=in_dim,
                    out_features=out_dim,
                    omega_0=self.first_omega_0 if layer_idx == 0 else self.hidden_omega_0,
                    is_first=(layer_idx == 0),
                    is_last=(layer_idx == len(dims)-2),
                    residual=residual,
                    norm_style=norm_style,
                    norm_mode=norm_mode,
                    dropout=dropout,
                    dropout_prob=dropout_prob,
                    weight_norm=weight_norm,
                )
            )
        self.network = nn.ModuleList(layers)

    def forward(self, se2_vec, x):
        points = x.clone().detach().requires_grad_(True)
        latent_vec = self.encoder(se2_vec)
        x = torch.cat([latent_vec, points], dim=1)
        inputs = x
        for layer_idx, layer in enumerate(self.network):
            if layer_idx+1 in self.latent_in:
                x = layer(x)
                x = torch.cat([x, inputs], dim=-1)
            else:
                x = layer(x)
        return x, latent_vec, points

    def get_latent(self, se2_vec):
        return self.encoder(se2_vec)

    def inference(self, x):
        inputs = x
        for layer_idx, layer in enumerate(self.network):
            if layer_idx+1 in self.latent_in:
                x = layer(x)
                x = torch.cat([x, inputs], dim=-1)
            else:
                x = layer(x)
        return x
