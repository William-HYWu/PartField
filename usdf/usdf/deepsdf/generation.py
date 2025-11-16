import numpy as np
import torch
from torch import nn, autograd
from tqdm import trange
from vedo import Plotter, Mesh, Points

import pytorch_kinematics as pk

from usdf.generation import BaseGenerator
from usdf.utils.infer_utils import inference_by_optimization
from usdf.utils.marching_cubes import create_mesh
import usdf.loss as usdf_losses


class RBF(torch.nn.Module):
    def __init__(self, sigma=None):
        super(RBF, self).__init__()
        self.sigma = sigma

    def forward(self, X, Y):
        XX = X.matmul(X.transpose(-2, -1))
        XY = X.matmul(Y.transpose(-2, -1))
        YY = Y.matmul(Y.transpose(-2, -1))

        XX_diag = torch.diagonal(XX, dim1=-2, dim2=-1).unsqueeze(-1)
        YY_diag = torch.diagonal(YY, dim1=-2, dim2=-1).unsqueeze(-2)
        dnorm2 = -2 * XY + XX_diag + YY_diag.transpose(-2, -1)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma = np.sqrt(h).item()
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY


def z_rot_to_matrix(rot_z_theta):
    size = np.array(list(rot_z_theta.shape) + [3, 3])
    size[:-2] = 1
    rot = torch.zeros_like(rot_z_theta).unsqueeze(-1).unsqueeze(-1).repeat(*size)
    rot[..., 0, 0] = torch.cos(rot_z_theta)
    rot[..., 0, 1] = -torch.sin(rot_z_theta)
    rot[..., 1, 0] = torch.sin(rot_z_theta)
    rot[..., 1, 1] = torch.cos(rot_z_theta)
    rot[..., 2, 2] = 1.0
    return rot


class Generator(BaseGenerator):

    def __init__(self, cfg: dict, model: nn.Module, generation_cfg: dict, device: torch.device = None):
        super().__init__(cfg, model, generation_cfg, device)

        self.generates_mesh = False
        self.generates_mesh_set = True

        self.gen_from_known_latent = generation_cfg.get("gen_from_known_latent", False)
        self.svgd = generation_cfg.get("svgd", False)
        self.batch_latent = generation_cfg.get("batch_latent", True)
        self.infer_pose = generation_cfg.get("infer_pose", True)
        self.pose_z_rot_only = generation_cfg.get("pose_z_rot_only", True)
        self.mesh_resolution = generation_cfg.get("mesh_resolution", 128)
        self.embed_weight = generation_cfg.get("embed_weight", 1.0)
        self.free_weight = generation_cfg.get("free_weight", 100.0)
        self.num_latent = generation_cfg.get("num_latent", 16)
        self.use_full_pointcloud = generation_cfg.get("use_full_pointcloud", False)
        self.init_mode = generation_cfg.get("init_mode", "random")
        self.outer_iter_limit = generation_cfg.get("outer_iter_limit", 5)
        self.iter_limit = generation_cfg.get("iter_limit", 100)
        self.vis_every = generation_cfg.get("vis_every", 250)
        self.vis_inter = generation_cfg.get("vis_inter", False)
        self.num_perturb = generation_cfg.get("num_perturb", 8)
        self.pos_sigma = generation_cfg.get("pos_sigma", 0.04)
        self.rot_sigma = generation_cfg.get("rot_sigma", 2.0)
        self.latent_sigma = generation_cfg.get("latent_sigma", 0.1)

    def generate_latent(self, data, return_single: bool = False):
        latent_metadata = {}
        if self.gen_from_known_latent:
            latent = self.model.encode_example(torch.tensor([data["example_idx"]]).to(self.device),
                                               torch.tensor([data["mesh_idx"]]).to(self.device), None)[0]
            pose = None
        else:
            if self.svgd:
                latent, pose, latent_metadata = self.infer_latent_svgd(data, num_examples=1)
            elif self.batch_latent:
                latent, pose, latent_metadata = self.infer_latent_batch(data, num_examples=1)
            else:
                latent, pose, latent_metadata = self.infer_latent(data, num_examples=1)

            if return_single:
                latent = latent[0, torch.argmin(latent_metadata["final_loss"][0])]
                pose = pose[0, torch.argmin(latent_metadata["final_loss"][0])]
            else:
                latent = latent[0]
                pose = pose[0]

        return (latent, pose), latent_metadata

    ####################################################################################################################
    # Inference Helpers                                                                                                #
    ####################################################################################################################

    def infer_latent_svgd(self, data_dict, num_examples=1):
        # Full point cloud.
        full_pointcloud = torch.from_numpy(data_dict["full_pointcloud"]).to(self.device).float().unsqueeze(0)
        full_pointcloud = full_pointcloud.repeat(num_examples, self.num_latent, 1, 1)
        full_pointcloud.requires_grad = True

        # Surface point cloud.
        surface_pointcloud = torch.from_numpy(data_dict["surface_pointcloud"]).to(self.device).float().unsqueeze(0)
        surface_pointcloud = surface_pointcloud.repeat(num_examples, self.num_latent, 1, 1)
        surface_pointcloud.requires_grad = True

        # Free point cloud.
        free_pointcloud = torch.from_numpy(data_dict["free_pointcloud"]).to(self.device).float().unsqueeze(0)
        free_pointcloud = free_pointcloud.repeat(num_examples, self.num_latent, 1, 1)
        free_pointcloud.requires_grad = True

        latent, pose = self.init_latent(num_examples)

        res_latent, res_pose, metadata = self.run_svgd(latent, pose, data_dict, surface_pointcloud, free_pointcloud,
                                                       full_pointcloud)
        res_loss = metadata["final_loss"]

        for iter_idx in range(self.outer_iter_limit):
            final_loss_sorted, indices = torch.sort(res_loss[0], descending=False)

            # Sort by loss.
            latent = res_latent[:, indices]
            pose = res_pose[:, indices]

            if self.vis_inter:
                self.vis_function(latent, pose, data_dict, final_loss=final_loss_sorted)

            # Choose the best result and create perturbations around it.
            best_latent = latent[:, 0:1]
            best_latent = best_latent.repeat(1, self.num_perturb, 1)
            best_pose = pose[:, 0:1]
            best_pose = best_pose.repeat(1, self.num_perturb, 1)

            # Update latents/poses.
            latent[:, self.num_latent - self.num_perturb:] = best_latent
            pose[:, self.num_latent - self.num_perturb:] = best_pose

            res_latent, res_pose, metadata = self.run_svgd(latent, pose, data_dict, surface_pointcloud,
                                                           free_pointcloud, full_pointcloud)
            res_loss = metadata["final_loss"]

        return res_latent, res_pose, {"final_loss": res_loss}

    def infer_latent(self, data_dict, num_examples=1):
        # Full point cloud.
        full_pointcloud = torch.from_numpy(data_dict["full_pointcloud"]).to(self.device).float().unsqueeze(0)
        full_pointcloud = full_pointcloud.repeat(num_examples, self.num_latent, 1, 1)
        full_pointcloud.requires_grad = True

        # Surface point cloud.
        surface_pointcloud = torch.from_numpy(data_dict["surface_pointcloud"]).to(self.device).float().unsqueeze(0)
        surface_pointcloud = surface_pointcloud.repeat(num_examples, self.num_latent, 1, 1)
        surface_pointcloud.requires_grad = True

        # Free point cloud.
        free_pointcloud = torch.from_numpy(data_dict["free_pointcloud"]).to(self.device).float().unsqueeze(0)
        free_pointcloud = free_pointcloud.repeat(num_examples, self.num_latent, 1, 1)
        free_pointcloud.requires_grad = True

        latent, pose = self.init_latent(num_examples)

        if self.infer_pose:
            res_latent, res_pose, metadata = self.run_sgd(latent, pose, data_dict, surface_pointcloud, free_pointcloud,
                                                          full_pointcloud)
            res_loss = metadata["final_loss"]

            for iter_idx in range(self.outer_iter_limit):
                final_loss_sorted, indices = torch.sort(res_loss[0], descending=False)

                # Sort by loss.
                latent = res_latent[:, indices]
                pose = res_pose[:, indices]

                if self.vis_inter:
                    self.vis_function(latent, pose, data_dict, final_loss=final_loss_sorted)

                # Choose the best result and create perturbations around it.
                best_latent = latent[:, 0:1]
                best_latent = best_latent.repeat(1, self.num_perturb, 1)
                best_pose = pose[:, 0:1]
                best_pose = best_pose.repeat(1, self.num_perturb, 1)

                # Update latents/poses.
                latent[:, self.num_latent - self.num_perturb:] = best_latent
                pose[:, self.num_latent - self.num_perturb:] = best_pose

                # Calculate perturbations across ALL values..
                delta_pos = torch.randn([num_examples, self.num_latent, 3], dtype=torch.float32,
                                        device=self.device) * self.pos_sigma
                delta_theta = torch.randn([num_examples, self.num_latent], dtype=torch.float32,
                                          device=self.device) * self.rot_sigma
                rand_dir = torch.randn([num_examples, self.num_latent, 3], dtype=torch.float32, device=self.device)
                rand_dir = rand_dir / torch.norm(rand_dir, dim=-1, keepdim=True)
                delta_rot = pk.axis_and_angle_to_matrix_33(rand_dir, delta_theta)

                # Combine with best result from previous iteration.
                pos = pose[:, :, :3]
                if self.pose_z_rot_only:
                    rot_z_theta = pose[..., 3]
                    rot_matrix = z_rot_to_matrix(rot_z_theta)
                else:
                    rot_6d = pose[:, :, 3:]
                    rot_matrix = pk.rotation_6d_to_matrix(rot_6d)
                new_rot_matrix = delta_rot @ rot_matrix
                if self.pose_z_rot_only:
                    new_rot_theta = torch.atan2(new_rot_matrix[..., 1, 0], new_rot_matrix[..., 0, 0]).unsqueeze(-1)
                else:
                    new_rot_theta = pk.matrix_to_rotation_6d(new_rot_matrix)
                new_pos = pos + delta_pos
                new_pose = torch.cat([new_pos, new_rot_theta], dim=-1)

                # TODO: Add offsets to latent code?
                latent_offsets = torch.randn([num_examples, self.num_latent, self.model.z_object_size],
                                             dtype=torch.float32, device=self.device) * self.latent_sigma
                new_latent = latent + latent_offsets

                if self.vis_inter:
                    self.vis_function(new_latent, new_pose, data_dict)

                latent, pose, metadata = self.run_sgd(new_latent, new_pose, data_dict, surface_pointcloud,
                                                      free_pointcloud,
                                                      full_pointcloud)
                final_loss = metadata["final_loss"]

                # Get best num_latent from current best and new results.
                latent = torch.cat([latent, res_latent], dim=1)
                pose = torch.cat([pose, res_pose], dim=1)
                final_loss = torch.cat([final_loss, res_loss], dim=1)

                final_loss_sorted, indices = torch.sort(final_loss[0], descending=False)
                res_latent = latent[:, indices[:self.num_latent]]
                res_pose = pose[:, indices[:self.num_latent]]
                res_loss = final_loss[:, indices[:self.num_latent]]
        else:
            res_latent, res_pose, metadata = self.run_sgd(latent, pose, data_dict, surface_pointcloud, free_pointcloud,
                                                          full_pointcloud)
            res_loss = metadata["final_loss"]

        return res_latent, res_pose, {"final_loss": res_loss}

    def init_latent_batch(self, num_examples):
        latent_init = torch.randn([num_examples, 4, self.model.z_object_size], dtype=torch.float32,
                                  device=self.device) * 0.1
        latent_init = torch.repeat_interleave(latent_init, 4, 1)

        pos = torch.zeros(3)
        if self.infer_pose:
            # rot_batch = pk.matrix_to_rotation_6d(
            #     torch.tile(torch.eye(3), (num_examples * self.num_latent, 1, 1))
            # ).to(self.device).reshape([num_examples, self.num_latent, 6])
            if self.pose_z_rot_only:
                rot_batch = torch.rand([num_examples, self.num_latent, 1], dtype=torch.float32,
                                       device=self.device) * 2 * np.pi
            else:
                rot_batch = pk.matrix_to_rotation_6d(
                    pk.random_rotations(num_examples * self.num_latent)).to(self.device).reshape(
                    [num_examples, self.num_latent, 6])
            pos_batch = torch.from_numpy(np.tile(pos, (num_examples, self.num_latent, 1))).to(self.device).float()
            pose_init = torch.cat([pos_batch, rot_batch], dim=-1)
        else:
            rot_batch = pk.matrix_to_rotation_6d(
                torch.tile(torch.eye(3), (num_examples * self.num_latent, 1, 1))
            ).to(self.device).reshape([num_examples, self.num_latent, 6])
            pos_batch = torch.from_numpy(np.tile(pos, (num_examples, self.num_latent, 1))).to(self.device).float()
            pose_init = torch.cat([pos_batch, rot_batch], dim=-1)

        return latent_init, pose_init

    def infer_latent_batch(self, data_dict, num_examples=1):
        # Full point cloud.
        full_pointcloud = torch.from_numpy(data_dict["full_pointcloud"]).to(self.device).float().unsqueeze(0)
        full_pointcloud = full_pointcloud.repeat(num_examples, self.num_latent, 1, 1)
        full_pointcloud.requires_grad = True

        # Surface point cloud.
        surface_pointcloud = torch.from_numpy(data_dict["surface_pointcloud"]).to(self.device).float().unsqueeze(0)
        surface_pointcloud = surface_pointcloud.repeat(num_examples, self.num_latent, 1, 1)
        surface_pointcloud.requires_grad = True

        # Free point cloud.
        free_pointcloud = torch.from_numpy(data_dict["free_pointcloud"]).to(self.device).float().unsqueeze(0)
        free_pointcloud = free_pointcloud.repeat(num_examples, self.num_latent, 1, 1)
        free_pointcloud.requires_grad = True

        latent, pose = self.init_latent_batch(num_examples)

        if self.infer_pose:
            res_latent, res_pose, metadata = self.run_sgd(latent, pose, data_dict, surface_pointcloud, free_pointcloud,
                                                          full_pointcloud)
            res_loss = metadata["final_loss"]

            for iter_idx in range(self.outer_iter_limit):
                latent_next = torch.zeros_like(res_latent)
                pose_next = torch.zeros_like(res_pose)
                for latent_set in range(4):
                    # Compute best latent for each set.
                    loss_set = res_loss[:, 4 * latent_set:4 * latent_set + 4]
                    best_loss, best_idx = torch.min(loss_set, dim=1)
                    latent_next[:, 4 * latent_set:4 * latent_set + 4] = (
                        res_latent[:, latent_set * 4 + best_idx.item()].unsqueeze(1).repeat(1, 4, 1))
                    pose_next[:, 4 * latent_set:4 * latent_set + 4] = (
                        res_pose[:, latent_set * 4 + best_idx.item()].unsqueeze(1).repeat(1, 4, 1))

                # Calculate perturbations across ALL values..
                delta_pos = torch.randn([num_examples, self.num_latent, 3], dtype=torch.float32,
                                        device=self.device) * self.pos_sigma
                delta_theta = torch.randn([num_examples, self.num_latent], dtype=torch.float32,
                                          device=self.device) * self.rot_sigma
                rand_dir = torch.randn([num_examples, self.num_latent, 3], dtype=torch.float32, device=self.device)
                rand_dir = rand_dir / torch.norm(rand_dir, dim=-1, keepdim=True)
                delta_rot = pk.axis_and_angle_to_matrix_33(rand_dir, delta_theta)

                # Combine with best result from previous iteration.
                pos = pose_next[:, :, :3]
                if self.pose_z_rot_only:
                    rot_z_theta = pose_next[..., 3]
                    rot_matrix = z_rot_to_matrix(rot_z_theta)
                else:
                    rot_6d = pose_next[:, :, 3:]
                    rot_matrix = pk.rotation_6d_to_matrix(rot_6d)
                new_rot_matrix = delta_rot @ rot_matrix
                if self.pose_z_rot_only:
                    new_rot_theta = torch.atan2(new_rot_matrix[..., 1, 0], new_rot_matrix[..., 0, 0]).unsqueeze(-1)
                else:
                    new_rot_theta = pk.matrix_to_rotation_6d(new_rot_matrix)
                new_pos = pos + delta_pos
                new_pose = torch.cat([new_pos, new_rot_theta], dim=-1)

                # TODO: Add offsets to latent code?
                latent_offsets = torch.randn([num_examples, self.num_latent, self.model.z_object_size],
                                             dtype=torch.float32, device=self.device) * self.latent_sigma
                new_latent = latent_next + latent_offsets

                if self.vis_inter:
                    self.vis_function(new_latent, new_pose, data_dict)

                latent, pose, metadata = self.run_sgd(new_latent, new_pose, data_dict, surface_pointcloud,
                                                      free_pointcloud,
                                                      full_pointcloud)
                final_loss = metadata["final_loss"]

                # Get best num_latent from current best and new results.
                new_res_latent = torch.zeros_like(res_latent)
                new_res_pose = torch.zeros_like(res_pose)
                new_res_loss = torch.zeros_like(res_loss)
                for latent_set in range(4):
                    # Combine set examples.
                    latent_combined = torch.cat([latent[:, latent_set * 4:latent_set * 4 + 4],
                                                 res_latent[:, latent_set * 4:latent_set * 4 + 4]], dim=1)
                    pose_combined = torch.cat([pose[:, latent_set * 4:latent_set * 4 + 4],
                                               res_pose[:, latent_set * 4:latent_set * 4 + 4]], dim=1)
                    loss_combined = torch.cat([final_loss[:, latent_set * 4:latent_set * 4 + 4],
                                               res_loss[:, latent_set * 4:latent_set * 4 + 4]], dim=1)

                    # Sort by loss.
                    loss_combined_sorted, indices = torch.sort(loss_combined[0], descending=False)

                    # Get best 4 from current best and new results.
                    new_res_latent[:, latent_set * 4:latent_set * 4 + 4] = latent_combined[:, indices[:4]]
                    new_res_pose[:, latent_set * 4:latent_set * 4 + 4] = pose_combined[:, indices[:4]]
                    new_res_loss[:, latent_set * 4:latent_set * 4 + 4] = loss_combined[:, indices[:4]]
                res_latent = new_res_latent
                res_pose = new_res_pose
                res_loss = new_res_loss
        else:
            res_latent, res_pose, metadata = self.run_sgd(latent, pose, data_dict, surface_pointcloud, free_pointcloud,
                                                          full_pointcloud)
            res_loss = metadata["final_loss"]

        return res_latent, res_pose, {"final_loss": res_loss}

    def run_sgd(self, latent, pose, data_dict, surface_pointcloud, free_pointcloud, full_pointcloud):
        latent = latent.detach()
        pose = pose.detach()

        # Set up as parameters for opt.
        latent.requires_grad = True
        pose.requires_grad = True

        if self.infer_pose:
            opt = torch.optim.Adam([latent, pose], lr=3e-2)
        else:
            opt = torch.optim.Adam([latent], lr=3e-2)

        iter_idx = 0
        range_ = trange(self.iter_limit)
        for iter_idx in range_:
            opt.zero_grad()

            if iter_idx % self.vis_every == 0 and False:
                self.vis_function(latent, pose, data_dict)

            loss, loss_ind = self.inference_loss(latent, pose,
                                                 full_pointcloud if self.use_full_pointcloud else surface_pointcloud,
                                                 free_pointcloud)

            loss.backward()
            opt.step()

            range_.set_postfix(loss=loss.item())

        _, final_loss = self.inference_loss(latent, pose,
                                            full_pointcloud if self.use_full_pointcloud else surface_pointcloud,
                                            free_pointcloud)

        return latent, pose, {"final_loss": final_loss, "iters": iter_idx + 1}

    #################################################################
    # SVGD stuff.
    #################################################################

    def log_prob(self, X, data_dict, surface_pointcloud, free_pointcloud, full_pointcloud):
        pose_size = 4 if self.pose_z_rot_only else 9
        pose = X[:, :, :pose_size]
        latent = X[:, :, pose_size:]

        loss, loss_ind = self.inference_loss(latent, pose,
                                             full_pointcloud if self.use_full_pointcloud else surface_pointcloud,
                                             free_pointcloud)
        return -loss

    def run_svgd(self, latent, pose, data_dict, surface_pointcloud, free_pointcloud, full_pointcloud):
        latent = latent.detach()
        pose = pose.detach()

        K = RBF()

        # Set up as parameters for opt.
        X = torch.cat([pose, latent], dim=-1)
        X.detach()
        X.requires_grad = True

        opt = torch.optim.Adam([X], lr=3e-2)

        iter_idx = 0
        range_ = trange(self.iter_limit)
        for iter_idx in range_:
            opt.zero_grad()

            X = X.detach().requires_grad_(True)

            log_prob = self.log_prob(X, data_dict, surface_pointcloud, free_pointcloud, full_pointcloud)
            score_func = autograd.grad(log_prob.sum(), X)[0]

            K_XX = K(X, X.detach())
            grad_K = -autograd.grad(K_XX.sum(), X)[0]

            phi = (K_XX.detach() @ score_func + grad_K) / X.shape[0]

            X.grad = -phi
            opt.step()

            range_.set_postfix(loss=log_prob.item())

        pose_size = 4 if self.pose_z_rot_only else 9
        latent = X[:, :, pose_size:]
        pose = X[:, :, :pose_size]

        _, final_loss = self.inference_loss(latent, pose,
                                            full_pointcloud if self.use_full_pointcloud else surface_pointcloud,
                                            free_pointcloud)

        return latent, pose, {"final_loss": final_loss, "iters": iter_idx + 1}

    #################################################################

    def init_latent(self, num_examples):
        # latent_init = torch.zeros([num_examples, self.num_latent, self.model.z_object_size], dtype=torch.float32,
        #                           device=self.device)
        # torch.nn.init.normal_(latent_init[..., 9:], mean=0.0, std=0.1)
        # latent_init = self.model.object_code.weight[0].unsqueeze(0).repeat(num_examples, self.num_latent, 1)
        latent_init = torch.randn([num_examples, self.num_latent, self.model.z_object_size], dtype=torch.float32,
                                  device=self.device) * 0.1

        pos = torch.zeros(3)
        if self.infer_pose:
            # rot_batch = pk.matrix_to_rotation_6d(
            #     torch.tile(torch.eye(3), (num_examples * self.num_latent, 1, 1))
            # ).to(self.device).reshape([num_examples, self.num_latent, 6])
            if self.pose_z_rot_only:
                rot_batch = torch.rand([num_examples, self.num_latent, 1], dtype=torch.float32,
                                       device=self.device) * 2 * np.pi
            else:
                rot_batch = pk.matrix_to_rotation_6d(
                    pk.random_rotations(num_examples * self.num_latent)).to(self.device).reshape(
                    [num_examples, self.num_latent, 6])
            pos_batch = torch.from_numpy(np.tile(pos, (num_examples, self.num_latent, 1))).to(self.device).float()
            pose_init = torch.cat([pos_batch, rot_batch], dim=-1)
        else:
            rot_batch = pk.matrix_to_rotation_6d(
                torch.tile(torch.eye(3), (num_examples * self.num_latent, 1, 1))
            ).to(self.device).reshape([num_examples, self.num_latent, 6])
            pos_batch = torch.from_numpy(np.tile(pos, (num_examples, self.num_latent, 1))).to(self.device).float()
            pose_init = torch.cat([pos_batch, rot_batch], dim=-1)

        return latent_init, pose_init

    def inference_loss(self, latent, pose, surface_pointcloud, free_pointcloud):
        # Pull out pose.
        pos = pose[..., :3]
        if self.pose_z_rot_only:
            rot_z_theta = pose[..., 3]
            rot = z_rot_to_matrix(rot_z_theta)
        else:
            rot_6d = pose[..., 3:]
            rot = pk.rotation_6d_to_matrix(rot_6d)

        # Transform points.
        surface_pointcloud_tf = ((rot @ surface_pointcloud.transpose(3, 2)).transpose(2, 3) +
                                 pos.unsqueeze(2).repeat(1, 1, surface_pointcloud.shape[2], 1))
        free_pointcloud_tf = ((rot @ free_pointcloud.transpose(3, 2)).transpose(2, 3) +
                              pos.unsqueeze(2).repeat(1, 1, free_pointcloud.shape[2], 1))

        # Predict with updated latents.
        surface_pred_dict = self.model.forward(surface_pointcloud_tf, latent)
        free_pred_dict = self.model.forward(free_pointcloud_tf, latent)

        # Loss: all points on surface should have SDF = 0.0.
        epsilon = 3.5e-4
        surface_loss = torch.mean(
            torch.max(torch.abs(surface_pred_dict["sdf"]) - epsilon, torch.zeros_like(surface_pred_dict["sdf"])),
            dim=-1)
        # surface_loss = torch.mean(
        #     torch.max(torch.zeros_like(surface_pred_dict["sdf"]), self.alpha + surface_pred_dict["sdf"]), dim=-1)

        # Loss: all points in free space should have SDF > 0.0.
        free_loss = torch.mean(torch.abs(torch.min(torch.zeros_like(free_pred_dict["sdf"]), free_pred_dict["sdf"])),
                               dim=-1)

        # Latent embedding loss: shouldn't drift too far from data.
        embedding_loss = usdf_losses.l2_loss(latent, squared=True, reduce=False)

        loss = surface_loss + self.free_weight * free_loss + self.embed_weight * embedding_loss
        return loss.mean(), loss

    def vis_function(self, latent, pose, data_dict, final_loss=None):
        meshes = []
        for mesh_idx in range(self.num_latent):
            mesh, _ = self.generate_mesh_from_latent(latent[0, mesh_idx], pose[0, mesh_idx])
            meshes.append(mesh)

        plot_shape = int(np.ceil(np.sqrt(self.num_latent)))
        plt = Plotter(shape=(plot_shape, plot_shape))
        for mesh_idx in range(self.num_latent):
            mesh = meshes[mesh_idx]
            plot_x = mesh_idx // plot_shape
            plot_y = mesh_idx % plot_shape
            plt.at(plot_x, plot_y).show(
                f"Mesh {mesh_idx}: Loss: {final_loss[mesh_idx].item():.4f}" if final_loss is not None else f"Mesh {mesh_idx}",
                Mesh([mesh.vertices, mesh.faces]),
                Points(data_dict["full_pointcloud"] if self.use_full_pointcloud else data_dict["surface_pointcloud"],
                       c="b"),
                Points(data_dict["free_pointcloud"], c="r", alpha=0.05),
            )
        plt.interactive().close()

    ####################################################################################################################

    def generate_mesh(self, data, metadata):
        # Generate a single latent code for the given data.
        l, _ = self.generate_latent(data, True)
        latent, pose = l

        # Generate mesh from latent code.
        return self.generate_mesh_from_latent(latent, pose)

    def generate_mesh_from_latent(self, latent, pose):
        latent = latent.unsqueeze(0)
        if pose is not None:
            pose = pose.unsqueeze(0)

        # Setup function to map from query points to SDF values.
        def sdf_fn(query_points):
            query_points = query_points.unsqueeze(0)

            if pose is not None:
                # Pull out pose.
                pos = pose[..., :3]
                if self.pose_z_rot_only:
                    rot_z_theta = pose[..., 3]
                    rot = z_rot_to_matrix(rot_z_theta)
                else:
                    rot_6d = pose[..., 3:]
                    rot = pk.rotation_6d_to_matrix(rot_6d)

                # Transform points.
                query_points = (rot @ query_points.transpose(1, 2)).transpose(2, 1) + pos

            return self.model.forward(query_points, latent)["sdf"][0]

        mesh = create_mesh(sdf_fn, n=self.mesh_resolution)
        return mesh, {"latent": latent, "pose": pose}

    def generate_mesh_set(self, data, metadata):
        # Generate a single latent code for the given data.
        l, l_metadata = self.generate_latent(data, False)
        latent, pose = l

        # Generate meshes from latent code.
        meshes = []
        for mesh_idx in range(self.num_latent):
            mesh, _ = self.generate_mesh_from_latent(latent[mesh_idx], pose[mesh_idx])
            meshes.append(mesh)

        return meshes, {"latent": latent, "pose": pose, "final_loss": l_metadata["final_loss"]}
