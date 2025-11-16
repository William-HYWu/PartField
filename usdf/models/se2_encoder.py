import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Sine_Layer(nn.Module):
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

class SE2Encoder(nn.Module):
    def __init__(self, mode, latent_size=32, hidden_dim=64, num_layers=3):
        super(SE2Encoder, self).__init__()
        layers = []
        input_dim = 4  # x, y, sin(theta), cos(theta)
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else latent_size
            if mode == 'sine':
                layers.append(Sine_Layer(input_dim, out_dim, is_first=(i==0)))
            else:
                layers.append(nn.Linear(input_dim, out_dim))
                if i < num_layers - 1:
                    layers.append(nn.ReLU())
            input_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, se2):
        return self.net(se2)