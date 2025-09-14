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


def get_reward(pred_qpos, target_pos, robot, end_effector, dofs_idx_local):
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
    reward = -torch.norm(joint_error)

    # End effector distance error bonus
    ee_error = target_pos - ee_pos
    task_space_bonus = -torch.norm(torch.tensor(ee_error))

    # Weighted combination 
    total_reward = reward + 0.1 * task_space_bonus

    return total_reward.item()

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

def train_ik_net_curriculum(vis=False, max_buffer_size=1000, save_path="checkpoints/ik_model_rl.pt"):
    wandb.init(project="veritas_v3", config={
        "lr": 1e-3,
        "action_std": 0.05,
        "initial_buffer_size": 1,
    })

    # Initialize Genesis
    gs.init(backend=gs.gpu, logging_level='warning',)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -3.5, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=60,
        ),
        show_viewer=vis,
    )

    scene.add_entity(gs.morphs.Plane())
    panda = scene.add_entity(
        gs.morphs.URDF(file="urdf/panda_bullet/panda_nohand.urdf", fixed=True)
    )
    target_marker = scene.add_entity(
        gs.morphs.Mesh(file="meshes/axis.obj", scale=0.2),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )

    scene.build()

    end_effector = panda.get_link("link7")
    joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    dofs_idx_local = [panda.get_joint(name).dofs_idx_local[0] for name in joints]

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

    # Curriculum Buffer
    buffer = add_point_to_buffer(workspace=workspace)
    
    reward_window = deque(maxlen=20)
    
    wandb.log({
                "buffer_targets": wandb.Table(
                    columns=["x", "y", "z"],
                    data=[buffer[-1].tolist()]
                )
            })

    step = 0
    cycle = 0
    while len(buffer) < max_buffer_size:
        for _ in range(len(buffer)):
            target_pos = buffer[np.random.randint(0, len(buffer))]
            qpos_mean = model(target_pos)
            distribution = dist_fn(qpos_mean, action_std)
            action = distribution.sample()
            log_prob = distribution.log_prob(action).sum()
            reward = get_reward(action, target_pos, panda, end_effector, dofs_idx_local)

            loss = -log_prob * reward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            reward_window.append(reward)
            
            wandb.log({
                "step": step,
                "reward": reward,
                "buffer_size": len(buffer),
                "loss": loss.item(),
                "reward_threshold": compute_reward_threshold(len(buffer), max_buffer_size=max_buffer_size, base_thresh=0.1, max_thresh=1.0),
            })
            
            step += 1

            
        # Curriculum step
        avg_reward = np.mean(reward_window) if reward_window else -np.inf
        
        wandb.log({
                "step": step,
                "cycle": cycle,
                "avg_reward": avg_reward,
                "buffer_size": len(buffer),
                "reward_threshold": compute_reward_threshold(len(buffer), max_buffer_size=max_buffer_size, base_thresh=0.1, max_thresh=1.0),
            })
        

        print(f"[Step {step:04d}] Buffer: {len(buffer)} | Reward: {avg_reward:.4f}")  
        
        cycle += 1
        
        if avg_reward > compute_reward_threshold(len(buffer), max_buffer_size=max_buffer_size, base_thresh=0.1, max_thresh=1.0):
            add_point_to_buffer(workspace=workspace, buffer=buffer, step_size=sample_distance)
            save_model(model, save_path)
            save_buffer(buffer)
            print(f"> Curriculum advanced. New buffer size: {len(buffer)}")
            wandb.log({"curriculum_advanced": step, "avg_reward": avg_reward})
            wandb.log({
                "buffer_targets": wandb.Table(
                    columns=["x", "y", "z"],
                    data=[buffer[-1].tolist()]
                )
            })
            
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    save_buffer(buffer)
    print(f"\nModel saved to: {save_path}")
    wandb.save(save_path)
    wandb.finish()

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to: {path}")

if __name__ == "__main__":
    train_ik_net_curriculum(vis=False, max_buffer_size=3000, save_path="checkpoints/ik_model.pt")
