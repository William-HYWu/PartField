from usdf.deepsdf.models.deepsdf import DeepSDF
from usdf.deepsdf.generation import Generator
from usdf.deepsdf.training import Trainer


def get_model(cfg, dataset, device=None):
    use_angle = cfg["model"].get("use_angle", False)
    use_pose_code = cfg["model"].get("use_pose_code", False)
    deep_sdf = DeepSDF(len(dataset), cfg["model"]["z_object_size"], use_angle=use_angle,
                       use_pose_code=use_pose_code, device=device)
    return deep_sdf


def get_trainer(cfg, model, device=None):
    trainer = Trainer(cfg, model, device)
    return trainer


def get_generator(cfg, model, generation_cfg, device=None):
    generator = Generator(cfg, model, generation_cfg, device)
    return generator
