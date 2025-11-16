import torch
import torch.nn as nn


class BaseGenerator(object):
    """
    Base generator class.

    Generator is responsible for implementing an API generating the shared representations from
    various model classes. Not all representations need to necessarily be implemented.
    """

    def __init__(self, cfg: dict, model: nn.Module, generation_cfg: dict, device: torch.device = None):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.generation_cfg = generation_cfg

        self.generates_mesh = False
        self.generates_mesh_set = False

    def generate_mesh(self, data, metadata):
        """
        Generate a single reconstruction (as mesh) from the given data and metadata.
        """
        raise NotImplementedError()

    def generate_mesh_set(self, data, metadata):
        """
        Generate a set of reconstructions (as meshes) where each mesh should represent a plausible
        reconstruction of the given data.
        """
        raise NotImplementedError()
