from torch import nn
import torch

from usdf.models import mlp


class DeepSDFObjectModule(nn.Module):

    def __init__(self, z_object_size: int, hidden_size: int = 512, out_dim: int = 1, final_activation: str = "none"):
        super().__init__()
        self.z_object_size = z_object_size
        self.final_activation = final_activation
        self.out_dim = out_dim

        self.object_module_stage1 = mlp.build_mlp(3 + self.z_object_size, hidden_size - (3 + self.z_object_size),
                                                  hidden_sizes=[hidden_size] * 3)
        self.object_module_stage2 = mlp.build_mlp(hidden_size, out_dim, hidden_sizes=[hidden_size] * 3)

    def forward(self, query_points: torch.Tensor, z_object: torch.Tensor):
        z_object_ = z_object.unsqueeze(1).repeat(1, query_points.shape[1], 1)
        model_in = torch.cat([query_points, z_object_], dim=-1)
        model_out = self.object_module_stage1(model_in)

        model_in_stage_2 = torch.cat([model_out, model_in], dim=-1)
        model_out = self.object_module_stage2(model_in_stage_2)

        if self.final_activation == "tanh":
            model_out = torch.tanh(model_out)

        if self.out_dim == 1:
            return model_out.squeeze(-1)
        return model_out
