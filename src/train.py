import torch
from torch import nn
from torch.optim import Adam
from ik_model import IKNet
from utils import add_point_to_buffer, IKWorkspace
import wandb
import genesis as gs
import os
import numpy as np
from collections import deque
import pickle
import math
from running_normalizer import RunningNormalizer
import time


def get_reward(pred_qpos, target_pos, robot, end_effector, dofs_idx_local, task_weight=1.0, joint_weight=0.1):
    # Get current pose
    robot.set_qpos(pred_qpos.detach().cpu().numpy())
    robot.scene.step()
    ee_pos = robot.get_link(end_effector.name).get_pos()

    # Get how ik would solve
    ik_qpos = robot.inverse_kinematics_multilink(
        links=[end_effector],
        poss=[target_pos],
        dofs_idx_local=dofs_idx_local,
    )

    # Joint angle error
    joint_error = pred_qpos - torch.tensor(ik_qpos, device=pred_qpos.device)

    # End effector distance error bonus
    target_pos = torch.tensor(target_pos, device=ee_pos.device, dtype=torch.float32)
    ee_error = torch.tensor(target_pos - ee_pos)

    # Weighted combination 
    reward = -task_weight * torch.norm(ee_error) - joint_weight * torch.norm(joint_error)

    return reward.item(), joint_error.norm().item(), ee_error.norm().item()

def compute_reward_threshold(buffer_size=1, max_buffer_size=1000, base_thresh=0.1, max_thresh=1.0):
    """ Algorithm to compute a dynamic reward threshold based on buffer size.
    Uses logarithmic scaling to prevent reward threshold from increasing extremely at the higher values.

    Args:
        buffer_size (int, optional): _description_. Defaults to 1.
        max_buffer_size (int, optional): _description_. Defaults to 1000.
        base_thresh (float, optional): Base reward threshold. Defaults to 0.1.
        max_thresh (float, optional): Maximum increase in thresh over base value. Defaults to 0.5.

    Returns:
        float: _description_
    """
    return -(base_thresh + ((math.log(buffer_size)/math.log(max_buffer_size))*(max_thresh-base_thresh)))

def train_ik_net(save_path="checkpoints/ik_model_rl.pt"):
    wandb.init(project="veritas_v4", config={
        "lr": 1e-3,
        "action_std": 0.05,
        "batch_size": 100,
        "avg_reward_complete": 0.1,
        "sample_distance": 0.05,
        "log_interval": 1,
    })
    
    

    # Initialize Genesis
    gs.init(backend=gs.gpu, logging_level='warning',)

    # Add robot to scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
    )
    panda = scene.add_entity(
        gs.morphs.URDF(file="urdf/panda_bullet/panda_nohand.urdf", fixed=True)
    )
    scene.build()

    # Get end effector and joints
    end_effector = panda.get_link("link7")
    joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    dofs_idx_local = [panda.get_joint(name).dofs_idx_local[0] for name in joints]

    # Initialize model, optimizer, and distribution
    model = IKNet()
    optimizer = Adam(model.parameters(), lr=wandb.config.lr)
    action_std = wandb.config.action_std
    dist_fn = torch.distributions.Normal

    # Generate workspace
    workspace = IKWorkspace()
    workspace.sphere_center = np.array([0.0, 0.0, 0.33])
    workspace.sphere_radius = 0.7
    workspace.cylinder_center = np.array([0.0, 0.0, 0.35])
    workspace.cylinder_radius = 0.14
    workspace.cylinder_height = 0.7 
    
    # Distance threshold for sampling
    sample_distance = 0.05

    # Normalizer
    reward_normalizer = RunningNormalizer()

    current_avg_reward = -np.inf
    
    step = 0
    cycle = 0
    while current_avg_reward < wandb.config.avg_reward_complete:
        
        # Generate batch data
        points = add_point_to_buffer(workspace=workspace)
        for _ in range(wandb.config.batch_size - 1):
            points = add_point_to_buffer(workspace=workspace, buffer=points, step_size=sample_distance)

        points = torch.stack(points)
        qpos_mean = model(points)
        distribution = dist_fn(qpos_mean, action_std)
        action = distribution.rsample()
        log_probs = distribution.log_prob(action).sum(dim=1)
        
        rewards = []
        joint_errors = []
        ee_errors = []
        
        for a, t in zip(action, points):
            r, je, ee = get_reward(a, t.cpu().numpy(), panda, end_effector, dofs_idx_local)
            rewards.append(r)
            joint_errors.append(je)
            ee_errors.append(ee)
            
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        reward_normalizer.update(rewards)
        norm_rewards = reward_normalizer.normalize(rewards)

        loss = -(log_probs * norm_rewards).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % wandb.config.log_interval == 0:
            wandb.log({
                "step": step,
                "loss": loss.item(),
                "avg_reward": rewards.mean().item(),
                "avg_joint_error": np.mean(joint_errors),
                "avg_ee_error": np.mean(ee_errors),
                "reward_min": rewards.min().item(),
                "reward_max": rewards.max().item(),
                "reward_std": rewards.std().item(),
                "reward_hist": wandb.Histogram(rewards.cpu().numpy()),
                "norm_reward_mean": norm_rewards.mean().item(),
                "norm_reward_std": norm_rewards.std().item(),
            })
        print(f"[{step:05d}] Reward: {rewards.mean():.3f} | Joint Err: {np.mean(joint_errors):.3f} | EE Err: {np.mean(ee_errors):.3f}")

        step += 1
        
        if(rewards.mean().item() > current_avg_reward):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"\nModel saved to: {save_path}")
            current_avg_reward = rewards.mean().item()
            
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
