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


def calculate_base_std(ee_error, max_error = 1.0, min_error = 0.01)->float:
    
    max_std = 0.5
    min_std = 0.05
    
    if ee_error is None:
        return max_std
    
    ee_error = ee_error.mean().item()
    if(ee_error >= max_error):
        return max_std
    if(ee_error <= min_error):
        return min_std
    
    # Normalize error to [0, 1] range
    t = (ee_error - min_error) / (max_error - min_error)
    t = max(0.0, min(1.0, t))  # clamp to [0, 1]

    # Interpolate between min and max std
    return min_std + t * (max_std - min_std)
    
    
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
    robot.control_dofs_position(qpos_mean.to(gs.device), envs_idx=envs_idx)
    robot.scene.step()

    jacobians = compute_geometric_jacobian(
        robot=robot,
        dofs_idx_local=dofs_idx_local,
        end_effector_link=end_effector_link,
        envs_idx=envs_idx
    )  # [B, 6, N_total]
    jacobians = jacobians[:, :, dofs_idx_local]                 # [B, 6, N]

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
    robot.control_dofs_position(pred_qpos.to(gs.device), envs_idx=envs_idx)

    # Step all environments in parallel
    robot.scene.step()

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
    
        # Compute reward components
        joint_error = torch.norm(pred_qpos - ik_qpos, dim=1)  # [B]

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

def pre_train_ik_net_curriculum(max_buffer_size=1000, save_path="checkpoints/ik_model_rl.pt"):
    wandb.init(project="veritas", config={
        "lr": 1e-3,
        "action_std": 0.1,
        "initial_buffer_size": 50,
        "avg_reward_complete": 0.1,
        "log_interval": 5,
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
    
    # Generate buffer for entire run
    buffer = generate_buffer(workspace, step_size=1, buffer_size=max_buffer_size, device="cuda")
    save_buffer(buffer)
    
    wandb.log({
        "buffer_targets": wandb.Table(
            columns=["x", "y", "z"],
            data=[item.tolist() for item in buffer]
        )
    })

    step = 0
    cycle = 0
    curriculum_advance_value = wandb.config.initial_buffer_size
    
    ee_error = None
    joint_error = None
    
    t0 = time.time()
    
    while curriculum_advance_value < max_buffer_size:
        target_pos = buffer[0:curriculum_advance_value]  # [N, 3]
        qpos_mean = model(target_pos)
        
        
        action_std = calculate_base_std(ee_error)
        
        per_joint_std = compute_per_joint_std(
            qpos_mean, 
            robot=panda, 
            dofs_idx_local=dofs_idx_local,
            end_effector_link=end_effector,
            base_std=action_std,
            normalize=True,
        )
        
        joint_std_per_joint = per_joint_std.mean(dim=0).tolist()  # [B]
        distribution = dist_fn(qpos_mean, per_joint_std)
        action = distribution.sample()
        rewards, ee_error, joint_error = get_reward_batch(action, target_pos, panda, end_effector, dofs_idx_local, use_angles=True, joint_weight=0.3)
        avg_reward = rewards.mean().item()
        
        log_probs = distribution.log_prob(action).sum(dim=1)  # [B]
        
        norm_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        loss = -(log_probs * norm_rewards).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        step += 1
        last_reward = avg_reward
        
        if step % wandb.config.log_interval == 0:
            wandb.log({
                "train/step": step,
                "train/loss": loss.item(),
                "train/avg_reward": avg_reward,
                "train/reward_std": rewards.std().item(),
                "train/reward_min": rewards.min().item(),
                "train/reward_max": rewards.max().item(),
                "train/log_prob_mean": log_probs.mean().item(),
                "train/buffer_size": curriculum_advance_value,
                "train/reward_threshold": compute_reward_threshold(
                    curriculum_advance_value, 
                    max_buffer_size=max_buffer_size, 
                    base_thresh=0.2, 
                    max_thresh=0.4
                ),
                "train/step_time_sec": (time.time() - t0) / wandb.config.log_interval,
                "train/ee_error_mean": ee_error.mean().item(),
                "train/joint_error_mean": joint_error.mean().item(),
                "train/action_std_joint_0": joint_std_per_joint[0],
                "train/action_std_joint_1": joint_std_per_joint[1],
                "train/action_std_joint_2": joint_std_per_joint[2],
                "train/action_std_joint_3": joint_std_per_joint[3],
                "train/action_std_joint_4": joint_std_per_joint[4],
                "train/action_std_joint_5": joint_std_per_joint[5],
                "train/action_std_joint_6": joint_std_per_joint[6],
            })
            t0 = time.time()
            
        if step % 100 == 0:
            wandb.log({
                "hist/target_positions": wandb.Histogram(target_pos.detach().cpu().numpy()),
                "hist/predicted_qpos": wandb.Histogram(qpos_mean.detach().cpu().numpy()),
            })

        print(f"[Step {step:04d}] Buffer: {curriculum_advance_value} | Reward: {avg_reward:.4f}")  
        
        
        if avg_reward > compute_reward_threshold(curriculum_advance_value, max_buffer_size=max_buffer_size, base_thresh=0.2, max_thresh=0.4):
            save_model(model, save_path)
            save_buffer(buffer)
            print(f"> Curriculum advanced. New buffer size: {curriculum_advance_value}")
            wandb.log({
                "curriculum/advanced_step": step,
                "curriculum/new_buffer_size": curriculum_advance_value,
                "curriculum/avg_reward": avg_reward,
            })
            curriculum_advance_value = curriculum_advance_value + 1
            
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    save_buffer(buffer)
    print(f"\nModel saved to: {save_path}")
    wandb.save(save_path)
    wandb.finish()

def post_train_ik_net(buffer_size=1000, 
                      save_path="checkpoints/ik_model_post.pt", 
                      max_steps=10000):
    """Training loop for post-training to squeeze out extra performance using a more robust optimization method (less exploration)

    Args:
        buffer_size (int, optional): _description_. Defaults to 1000.
        save_path (str, optional): _description_. Defaults to "checkpoints/ik_model_post.pt".
        max_steps (int, optional): _description_. Defaults to 10000.
    """
    wandb.init(project="veritas_posttrain", config={
        "lr": 1e-4,
        "action_std": 0.02,
        "max_steps": max_steps,
        "log_interval": 5,
    })

    # Initialize Genesis
    gs.init(backend=gs.gpu, 
            logging_level='warning', 
            # performance_mode=True
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

    scene.add_entity(gs.morphs.Plane())
    panda = scene.add_entity(
        gs.morphs.URDF(file="urdf/panda_bullet/panda_nohand.urdf", fixed=True)
    )

    scene.build(n_envs=buffer_size, env_spacing=(1.0, 1.0))

    end_effector = panda.get_link("link7")
    joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    dofs_idx_local = [panda.get_joint(name).dofs_idx_local[0] for name in joints]

    model = IKNet()
    
    # Optional: load pre-trained weights
    pretrain_path = "checkpoints/ik_model.pt"
    if os.path.exists(pretrain_path):
        model.load_state_dict(torch.load(pretrain_path))
        print(f"Loaded pretrained model from: {pretrain_path}")
    
    optimizer = Adam(model.parameters(), lr=wandb.config.lr)
    action_std = wandb.config.action_std
    dist_fn = torch.distributions.Normal

    # Generate new workspace and buffer
    workspace = IKWorkspace()
    workspace.sphere_center = np.array([0.0, 0.0, 0.33])
    workspace.sphere_radius = 0.7
    workspace.cylinder_center = np.array([0.0, 0.0, 0.35])
    workspace.cylinder_radius = 0.14
    workspace.cylinder_height = 0.7 

    buffer = generate_buffer(workspace, step_size=1, buffer_size=buffer_size, device="cuda")
    save_buffer(buffer, path="checkpoints/buffer_post.pkl")

    wandb.log({
        "posttrain_buffer_targets": wandb.Table(
            columns=["x", "y", "z"],
            data=[item.tolist() for item in buffer]
        )
    })

    best_reward = -float('inf')
    step = 0

    while step < wandb.config.max_steps:
        target_pos = buffer  # All points
        qpos_mean = model(target_pos)
        action_std = 0.5  # Much lower noise
        distribution = dist_fn(qpos_mean, action_std)
        action = distribution.rsample()  # Reparameterized for smoother gradients

        
        # Model should no longer need angles to optimize
        rewards, ee_error, joint_error = get_reward_batch(
            action, 
            target_pos, 
            panda, 
            end_effector, 
            dofs_idx_local, 
        )

        avg_reward = rewards.mean().item()

        log_probs = distribution.log_prob(action).sum(dim=1)
        loss = -(log_probs * rewards).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1

        if step % wandb.config.log_interval == 0:
            wandb.log({
                "step": step,
                "avg_reward": avg_reward,
                "best_reward": best_reward,
                "loss": loss.item()
            })
            print(f"[Step {step}] Avg Reward: {avg_reward:.4f} | Best: {best_reward:.4f}")

        # Save if performance improved
        if avg_reward > best_reward:
            best_reward = avg_reward
            save_model(model, save_path)
            print(f"> New best model saved with avg_reward: {avg_reward:.4f}")
            wandb.log({"best_model_saved_step": step, "avg_reward": avg_reward})

    wandb.finish()
    print(f"Finished post-training. Best reward: {best_reward:.4f}")

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to: {path}")

if __name__ == "__main__":
    pre_train_ik_net_curriculum(max_buffer_size=10000, save_path="checkpoints/ik_model.pt")
    # post_train_ik_net(buffer_size=10000, save_path="checkpoints/ik_model_post.pt")