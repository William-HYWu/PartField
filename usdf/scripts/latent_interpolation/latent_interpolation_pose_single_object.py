import os

import numpy as np
import torch
import yaml
from vedo import Plotter, Video, Mesh, settings
import vedo.pyplot as plt

from usdf import config
from usdf.utils.args_utils import get_model_dataset_arg_parser, load_model_dataset_from_args
from usdf.utils.model_utils import load_generation_cfg


def latent_interpolation(model_cfg, model, model_file, dataset, device, out_fn, object_idx, gen_args: dict, tsne_dir,
                         any_pose_code: bool = False):
    video_length = 5.0
    num_interpolations = 10
    settings.immediate_rendering = False  # Faster for multi-renderers.
    assert model.use_pose_code, "Model must use pose code for this script."
    assert object_idx < dataset.get_num_objects(), "Object idx must be less than number of objects."

    model.eval()

    # Load generate cfg, if present.
    generation_cfg = load_generation_cfg(model_cfg, model_file)
    if gen_args is not None:
        generation_cfg.update(gen_args)

    # Load generator.
    generator = config.get_generator(model_cfg, model, generation_cfg, device)

    # Load object codes/embeddings.
    tsne_object_codes = np.load(os.path.join(tsne_dir, "object_codes.npy"))
    object_codes = model.object_code.weight.detach().cpu().numpy()[:dataset.get_num_objects()]
    assert len(tsne_object_codes) == len(object_codes)

    # Load pose codes/embeddings.
    tsne_pose_codes = np.load(os.path.join(tsne_dir, "pose_codes.npy"))
    pose_codes = model.pose_code.weight.detach().cpu().numpy()
    assert len(tsne_pose_codes) == len(pose_codes)

    # Highlight pose codes for the object we're interpolating.
    if any_pose_code:
        tsne_pose_codes_obj = tsne_pose_codes
        pose_codes_obj = pose_codes
    else:
        tsne_pose_codes_obj = tsne_pose_codes[object_idx * dataset.N_transforms:(object_idx + 1) * dataset.N_transforms]
        pose_codes_obj = pose_codes[object_idx * dataset.N_transforms:(object_idx + 1) * dataset.N_transforms]

    # Visualize interpolation.
    plot_shape = [
        dict(bottomleft=(0.0, 0.0), topright=(0.5, 1.0)),
        dict(bottomleft=(0.5, 0.5), topright=(1.0, 1.0)),
        dict(bottomleft=(0.49 - 0.15, 0.99 - 0.2), topright=(0.49, 0.99)),
        dict(bottomleft=(0.5, 0.0), topright=(1.0, 0.5))
    ]
    vedo_plot = Plotter(shape=plot_shape, sharecam=False, size=(1920, 1080))
    vedo_plot.at(0).camera.SetPosition(1.5, 1.5, 1.0)
    vedo_plot.at(0).camera.SetFocalPoint(0.0, 0.0, 0.0)
    vedo_plot.at(0).camera.SetViewUp(0.0, 0.0, 1.0)
    vedo_plot.at(0).camera.SetClippingRange(0.01, 1000)
    vedo_plot.at(2).camera.SetPosition(-1.5, -1.5, -1.0)
    vedo_plot.at(2).camera.SetFocalPoint(0.0, 0.0, 0.0)
    vedo_plot.at(2).camera.SetViewUp(0.0, 0.0, 1.0)
    vedo_plot.at(2).camera.SetClippingRange(0.01, 1000)

    # Create plot of TSNE embeddings.
    tsne_plot = plt.plot(tsne_object_codes[:, 0], tsne_object_codes[:, 1], lw=0, marker=".", mc="blue", grid=False,
                         axes=False)
    vedo_plot.at(1).show(tsne_plot, "Object Embeddings")
    object_marker = plt.plot([tsne_object_codes[object_idx][0]], [tsne_object_codes[object_idx][1]], lw=0, marker="*",
                             mc="green", grid=False, like=tsne_plot, axes=False)
    vedo_plot.at(1).show(object_marker)

    pose_tsne_plot = plt.plot(tsne_pose_codes[:, 0], tsne_pose_codes[:, 1], lw=0, marker=".", mc="blue", grid=False,
                              axes=False)
    vedo_plot.at(3).show(pose_tsne_plot, "Pose Embeddings")
    pose_obj_tsne_plot = plt.plot(tsne_pose_codes_obj[:, 0], tsne_pose_codes_obj[:, 1], lw=0, marker="*", mc="red",
                                  grid=False, like=pose_tsne_plot, axes=False)
    vedo_plot.at(3).show(pose_obj_tsne_plot)

    video = Video(out_fn, backend="ffmpeg", fps=10)
    num_frames = int(video_length * video.fps)

    # Starting point for interpolation.
    pose_idx1 = np.random.choice(len(tsne_pose_codes_obj), size=1, replace=False)[0]

    pose_interp_marker = None
    vedo_mesh = None
    for interp_idx in range(num_interpolations):

        #############################
        # Interpolate in pose space #
        #############################

        # Select next embedding to interpolate to.
        pose_idx2 = np.random.choice(len(tsne_pose_codes_obj), size=1, replace=False)[0]

        # Draw line between embeddings.
        pose_interp_line = plt.plot([tsne_pose_codes_obj[pose_idx1, 0], tsne_pose_codes_obj[pose_idx2, 0]],
                                    [tsne_pose_codes_obj[pose_idx1, 1], tsne_pose_codes_obj[pose_idx2, 1]],
                                    like=pose_tsne_plot, lw=3, marker="o", mc="red", grid=False, axes=False)
        vedo_plot.at(3).show(pose_interp_line)

        for frame_idx in range(num_frames + 1):
            # Interpolate in tsne embeddings.
            pose_interp_tsne = tsne_pose_codes_obj[pose_idx1] * (1.0 - frame_idx / num_frames) + tsne_pose_codes_obj[
                pose_idx2] * (frame_idx / num_frames)

            # Plot interpolated tsne embedding.
            if pose_interp_marker is not None:
                vedo_plot.at(3).remove(pose_interp_marker)
            pose_interp_marker = plt.plot([pose_interp_tsne[0]], [pose_interp_tsne[1]], lw=0, marker="*", mc="green",
                                          grid=False, like=pose_tsne_plot, axes=False)
            vedo_plot.at(3).show(pose_interp_marker)

            # Generate mesh from interpolated embedding.
            interp_code = pose_codes_obj[pose_idx1] * (1.0 - frame_idx / num_frames) + pose_codes_obj[pose_idx2] * (
                    frame_idx / num_frames)
            object_code = object_codes[object_idx]
            latent = torch.from_numpy(np.concatenate([object_code, interp_code])).to(device).unsqueeze(0)
            mesh, _ = generator.generate_mesh({}, {"latent": latent})
            if vedo_mesh is not None:
                vedo_plot.at(0).remove(vedo_mesh)
                vedo_plot.at(2).remove(vedo_mesh)
            vedo_mesh = Mesh([mesh.vertices, mesh.faces])
            vedo_plot.at(0).show(vedo_mesh)
            vedo_plot.at(2).show(vedo_mesh)

            # Add frame to video.
            video.add_frame()

        # Remove pose interpolation line.
        vedo_plot.at(3).remove(pose_interp_line)

        #########################
        # Update starting point #
        #########################
        pose_idx1 = pose_idx2

    video.close()
    vedo_plot.close()


if __name__ == '__main__':
    parser = get_model_dataset_arg_parser()
    parser.add_argument("tsne_dir", type=str, help="Directory containing TSNE embeddings.")
    parser.add_argument("out_fn", type=str, help="Out filename for video.")
    parser.add_argument("object_idx", type=int, help="Object index to interpolate.")
    parser.add_argument("--gen_args", type=yaml.safe_load, default=None, help="Generation args.")
    parser.add_argument("--any_pose_code", action="store_true",
                        help="Use any pose code, not just those for the object.")
    parser.set_defaults(any_pose_code=False)
    args = parser.parse_args()

    model_cfg_, model_, dataset_, device_ = load_model_dataset_from_args(args)
    latent_interpolation(model_cfg_, model_, args.model_file, dataset_, device_, args.out_fn, args.object_idx,
                         args.gen_args, args.tsne_dir, args.any_pose_code)
