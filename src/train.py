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

@torch.no_grad()
def get_reward_batch(pred_qpos: torch.Tensor, 
                     target_pos: torch.Tensor, 
                     robot, 
                     end_effector, 
                     dofs_idx_local, 
                     task_weight=1.0, 
                     joint_weight=0.1,
                     use_angles=False) -> torch.Tensor:
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
        torch.Tensor: [B] reward for each sample.
    """
    B = pred_qpos.shape[0]
    envs_idx = list(range(B))
    
    # Apply predicted joint positions across all envs
    robot.control_dofs_position(pred_qpos.to(gs.device), envs_idx=envs_idx)

    # Step all environments in parallel
    robot.scene.step()

    # Get current EE positions for each env
    ee_link = robot.get_link(end_effector.name)
    ee_pos = ee_link.get_pos()[0:B]  # [B, 3]
        
    poss = target_pos.detach().cpu().numpy().reshape(1, -1, 3)  # This is the format the IK solver wants [1, B, 3]
    
    ee_error = torch.norm(target_pos - ee_pos[0:B], dim=1) # [B]
    
    if use_angles:
        # Batched IK solution
        ik_qpos = robot.inverse_kinematics_multilink(
                links= [ee_link],
                poss=poss,
                dofs_idx_local=dofs_idx_local,
                envs_idx=envs_idx,
        )

        ik_qpos = torch.tensor(ik_qpos, device=pred_qpos.device, dtype=pred_qpos.dtype)  # [B, 7]
    
        # Compute reward components
        joint_error = torch.norm(pred_qpos - ik_qpos, dim=1)  # [B]

        return -task_weight * ee_error - joint_weight * joint_error  # [B]
    else:
        return -task_weight * ee_error  # [B]
        

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
    wandb.init(project="veritas_v4", config={
        "lr": 1e-3,
        "action_std": 0.05,
        "initial_buffer_size": 50,
        "avg_reward_complete": 0.1,
        "log_interval": 5,
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

    scene.build(n_envs=max_buffer_size, env_spacing=(1.0, 1.0))

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

    # Generate buffer for entire run
    buffer = generate_buffer(workspace, step_size=1, buffer_size=max_buffer_size, device="cuda")
    save_buffer(buffer)

    reward_window = deque()
    
    wandb.log({
        "buffer_targets": wandb.Table(
            columns=["x", "y", "z"],
            data=[item.tolist() for item in buffer]
        )
    })

    step = 0
    cycle = 0
    curriculum_advance_value = wandb.config.initial_buffer_size
    while curriculum_advance_value < max_buffer_size:
        target_pos = buffer[0:curriculum_advance_value]  # [N, 3]
        qpos_mean = model(target_pos)
        distribution = dist_fn(qpos_mean, action_std)
        action = distribution.sample()
        rewards = get_reward_batch(action, target_pos, panda, end_effector, dofs_idx_local, use_angles=True)
        avg_reward = rewards.mean().item()
        
        log_probs = distribution.log_prob(action).sum(dim=1)  # [B]
        
        norm_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        loss = -(log_probs * norm_rewards).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        step += 1
        
        if step % wandb.config.log_interval == 0:
            wandb.log({
                    "step": step,
                    "avg_reward": avg_reward,
                    "buffer_size": curriculum_advance_value,
                    "reward_threshold": compute_reward_threshold(curriculum_advance_value, max_buffer_size=max_buffer_size, base_thresh=0.4, max_thresh=0.6),
                })

        print(f"[Step {step:04d}] Buffer: {curriculum_advance_value} | Reward: {avg_reward:.4f}")  
        
        
        if avg_reward > compute_reward_threshold(curriculum_advance_value, max_buffer_size=max_buffer_size, base_thresh=0.4, max_thresh=0.6):
            save_model(model, save_path)
            save_buffer(buffer)
            print(f"> Curriculum advanced. New buffer size: {curriculum_advance_value}")
            wandb.log({"curriculum_advanced": step, "avg_reward": avg_reward})
            curriculum_advance_value = curriculum_advance_value + 1
            
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
    train_ik_net_curriculum(vis=False, max_buffer_size=10000, save_path="checkpoints/ik_model.pt")