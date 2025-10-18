import torch
from torch import nn
from torch.optim import Adam
from ik_model import IKNet
from utils import generate_buffer, IKWorkspace
import wandb
import genesis as gs
import os
import numpy as np
from collections import deque
import pickle
import math
import time
from typing import Tuple
    
@torch.no_grad()
def compute_geometric_jacobian(robot, dofs_idx_local, end_effector_link, envs_idx):
    """
    Computes the geometric Jacobian [B, 6, N] for the given robot configuration.

    Args:
        robot: Genesis robot object
        dofs_idx_local (List[int]): Joint DOF indices
        end_effector_link: The end-effector link object
        envs_idx: List of env indices to compute for

    Returns:
        torch.Tensor: [B, 6, N] Jacobian
    """
    B = len(envs_idx)
    N = len(dofs_idx_local)

    # Allocate tensor for Jacobians
    jacobians = torch.zeros(B, 6, N, device=gs.device)

    # Get end-effector positions
    ee_pos = end_effector_link.get_pos()[envs_idx]         # [B, 3]

    for i, joint_idx in enumerate(dofs_idx_local):
        joint = robot.joints[joint_idx]
        joint_axis = joint.get_anchor_axis()[envs_idx]       # [B, 3] joint axis in world frame
        joint_pos = joint.get_anchor_pos()[envs_idx]              # [B, 3] joint position in world frame

        r = ee_pos - joint_pos                             # [B, 3]
        linear = torch.cross(joint_axis, r, dim=1)         # [B, 3]
        angular = joint_axis                               # [B, 3]

        jacobians[:, 0:3, i] = linear
        jacobians[:, 3:6, i] = angular

    return jacobians  # [B, 6, N]
    
@torch.no_grad()
def compute_per_joint_std(qpos_mean: torch.Tensor,
                          robot,
                          dofs_idx_local,
                          end_effector_link,
                          base_std=1.0,
                          normalize=True) -> torch.Tensor:
    """
    Compute per-joint standard deviation based on Jacobian magnitude.

    Args:
        qpos_mean (Tensor): [B, N] Mean joint configurations
        robot: Genesis robot instance
        dofs_idx_local (List[int]): Indices of joints to control
        end_effector_link: The end-effector link
        base_std (float): Base noise scaling factor
        normalize (bool): Whether to normalize joint sensitivities to [0, 1]

    Returns:
        Tensor: [B, N] Per-joint standard deviations
    """
    B, N = qpos_mean.shape
    envs_idx = list(range(B))

    # Set robot to qpos_mean
    robot.set_dofs_position(qpos_mean.to(gs.device), envs_idx=envs_idx, dofs_idx_local=dofs_idx_local, zero_velocity=True)

    jacobians = compute_geometric_jacobian(
        robot=robot,
        dofs_idx_local=dofs_idx_local,
        end_effector_link=end_effector_link,
        envs_idx=envs_idx
    )  # [B, 6, N_total]
    # jacobians = jacobians[:, :, dofs_idx_local]                 # [B, 6, N]

    # Compute joint sensitivity: L2 norm over spatial dimension (6D twist)
    joint_sensitivity = torch.norm(jacobians, dim=1)  # [B, N]

    if normalize:
        joint_sensitivity = joint_sensitivity / (joint_sensitivity.max(dim=1, keepdim=True)[0] + 1e-6)

    per_joint_std = base_std * joint_sensitivity
    return per_joint_std  # [B, N]


@torch.no_grad()
def get_reward_batch(pred_qpos: torch.Tensor, 
                     target_pos: torch.Tensor, 
                     robot, 
                     end_effector, 
                     dofs_idx_local, 
                     task_weight=1.0, 
                     joint_weight=0.1,
                     use_angles=False,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute batched reward for predicted joint positions. This function is very picky with parallelization so be careful of dimensions.

    Args:
        pred_qpos (torch.Tensor): [B, 7] predicted joint angles.
        target_pos (torch.Tensor): [B, 3] target end-effector positions.
        robot: Genesis robot instance.
        end_effector: End effector link object.
        dofs_idx_local (List[int]): DOF indices for the IK solver.
        task_weight (float): EE position error weight.
        joint_weight (float): IK difference penalty weight.
        use_angles (bool): Whether to use angular distance for joint error.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: [B] reward for each sample, and added error for end effector position and joint error.
    """
    B = pred_qpos.shape[0]
    envs_idx = list(range(B))
    
    # Apply predicted joint positions across all envs
    robot.set_dofs_position(pred_qpos.to(gs.device), envs_idx=envs_idx, dofs_idx_local=dofs_idx_local, zero_velocity=True)

    # Get current EE positions for each env
    ee_link = robot.get_link(end_effector.name)
    ee_pos = ee_link.get_pos()[0:B]  # [B, 3]
        
    poss = target_pos.detach().cpu().numpy().reshape(1, -1, 3)  # This is the format the IK solver wants [1, B, 3]
    
    ee_error = torch.norm(target_pos - ee_pos[0:B], dim=1) # [B]
    
    joint_error = torch.zeros_like(ee_error)
    
    if use_angles:
        # Batched IK solution
        ik_qpos = robot.inverse_kinematics_multilink(
                links= [ee_link],
                poss=poss,
                dofs_idx_local=dofs_idx_local,
                envs_idx=envs_idx,
        )

        ik_qpos = torch.tensor(ik_qpos, device=pred_qpos.device, dtype=pred_qpos.dtype)  # [B, 7]
        ik_qpos_subset = ik_qpos[:, dofs_idx_local]  # [B, 3]  
    
        # Compute reward components
        joint_error = torch.norm(pred_qpos - ik_qpos_subset, dim=1)  # [B]

    reward = -task_weight * ee_error - joint_weight * joint_error  # [B]
    
    return reward, ee_error, joint_error  # [B], [B], [B]
        

def save_buffer(buffer, path="checkpoints/buffer.pkl"):
    with open(path, 'wb') as f:
        pickle.dump(buffer, f)
    print(f"Buffer saved to: {path}")

def load_buffer(path="checkpoints/buffer.pkl"):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            buffer = pickle.load(f)
        print(f"Loaded buffer from: {path}")
        return buffer
    return []

def train_ik_net(max_iterations=5000, save_path="checkpoints/ik_model_rl.pt", workspace_path="ik_workspace.json"):
    wandb.init(project="veritas", config={
        "lr": 0.02,
        "buffer_size": 1000,
        "log_interval": 5,
        "entropy_weight": 0.1,
        "joint_weight": 0.2,
        "action_std": 0.02,
        "normalize_std": True,
        "use_angles": True,
        "task_weight": 1.0,
    })

    # Initialize Genesis
    gs.init(backend=gs.gpu, 
            logging_level='warning',
            # performance_mode=True,
            )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -3.5, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=60,
        ),
    )

    links_to_keep = ["LF_FOOT"]
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="anymal_c/urdf/anymal_c.urdf",
            fixed=True,
            collision=False,
            pos=(0, 0, 1),
            links_to_keep=links_to_keep,
        ),
    )

    buffer_size = wandb.config.buffer_size

    scene.build(n_envs=buffer_size, env_spacing=(1.0, 1.0))

    joints_name = ("LF_HAA", "LF_HFE", "LF_KFE")
    dofs_idx_local = [robot.get_joint(name).dofs_idx_local[0] for name in joints_name]
    print(f"Controlling joints: {joints_name} at DOF indices {dofs_idx_local}, total idx length: {len(dofs_idx_local)}")
    end_effector = robot.get_link("LF_FOOT")

    model = IKNet(output_dim=len(dofs_idx_local))
    optimizer = Adam(model.parameters(), lr=wandb.config.lr)
    dist_fn = torch.distributions.Normal

    # Generate workspace
    workspace = None
    try:
        workspace = IKWorkspace.load_from_file(workspace_path)
        print(f"Loaded workspace from {workspace_path}")
    except Exception as e:
        print(f"Failed to load workspace from {workspace_path}, using default. Error: {e}")
    
    # Generate buffer for entire run
    buffer = generate_buffer(workspace, step_size=1, buffer_size=buffer_size, device="cuda")
    save_buffer(buffer)
    
    wandb.log({
        "buffer_targets": wandb.Table(
            columns=["x", "y", "z"],
            data=[item.tolist() for item in buffer]
        )
    })

    step = 0
    
    ee_error = None
    joint_error = None
    
    t0 = time.time()
    
    joint_weight = wandb.config.joint_weight
    entropy_weight = wandb.config.entropy_weight
    action_std = wandb.config.action_std
    normalize_std = wandb.config.normalize_std
    use_angles = wandb.config.use_angles
    task_weight = wandb.config.task_weight
    
    while step < max_iterations:
        qpos_mean = model(buffer) # [N, 3]
            
        per_joint_std = compute_per_joint_std(
            qpos_mean, 
            robot=robot, 
            dofs_idx_local=dofs_idx_local,
            end_effector_link=end_effector,
            base_std=action_std,
            normalize=normalize_std,
        )
        
        joint_std_per_joint = per_joint_std.mean(dim=0).tolist()  # [B]
        distribution = dist_fn(qpos_mean, per_joint_std)
        action = distribution.sample()
        rewards, ee_error, joint_error = get_reward_batch(action, buffer, robot, end_effector, dofs_idx_local, use_angles=use_angles, task_weight=task_weight, joint_weight=joint_weight)
        avg_reward = rewards.mean().item()
        
        log_probs = distribution.log_prob(action).sum(dim=1)  # [B]
        
        entropy = distribution.entropy().sum(dim=1).mean()
        
        norm_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        loss = -(log_probs * norm_rewards).mean() - (entropy_weight * entropy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        step += 1
        
        if step % wandb.config.log_interval == 0:
            wandb.log({
                "train/step": step,
                "train/loss": loss.item(),
                "train/entropy": entropy.item(),
                "train/avg_reward": avg_reward,
                "train/reward_std": rewards.std().item(),
                "train/reward_min": rewards.min().item(),
                "train/reward_max": rewards.max().item(),
                "train/log_prob_mean": log_probs.mean().item(),
                "train/step_time_sec": (time.time() - t0) / wandb.config.log_interval,
                "train/ee_error_mean": ee_error.mean().item(),
                "train/joint_error_mean": joint_error.mean().item(),
                "train/action_std_joint_0": joint_std_per_joint[0],
                "train/action_std_joint_1": joint_std_per_joint[1],
                "train/action_std_joint_2": joint_std_per_joint[2],
            })
            t0 = time.time()

        print(f"[Step {step:04d}] Reward: {avg_reward:.4f}")  
            
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to: {save_path}")
    wandb.save(save_path)
    wandb.finish()

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to: {path}")

if __name__ == "__main__":
    train_ik_net(save_path="checkpoints/ik_model.pt")
