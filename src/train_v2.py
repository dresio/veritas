import torch
from torch import nn
from torch.optim import Adam
from ik_model import IKNet
from utils import sample_point
import wandb
import genesis as gs
import os
import numpy as np
from collections import deque

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


def train_ik_net_curriculum(vis=False, max_buffer_size=1000, save_path="checkpoints/ik_model_rl.pt"):
    wandb.init(project="veritas", config={
        "lr": 1e-3,
        "action_std": 0.05,
        "reward_threshold": -0.1,
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

    # Sampling params
    inner_radius = 0.14
    outer_radius = 0.7
    sphere_pos = (0, 0, 0.33)
    cylinder_pos = (0, 0, 0.35)
    cylinder_height = 0.7

    # Curriculum Buffer
    buffer = []
    buffer_size = wandb.config.initial_buffer_size
    reward_threshold = wandb.config.reward_threshold
    reward_window = deque(maxlen=20)

    def expand_buffer():
        new_target = sample_point(
            sphere_radius=outer_radius,
            sphere_center=sphere_pos,
            cylinder_radius=inner_radius,
            cylinder_height=cylinder_height,
            cylinder_center=cylinder_pos,
        )
        buffer.append(torch.tensor(new_target, dtype=torch.float32))

    # Fill initial buffer
    for _ in range(buffer_size):
        expand_buffer()

    step = 0
    while buffer_size < max_buffer_size:
        for _ in range(buffer_size):
            target_pos = buffer[np.random.randint(0, buffer_size)]
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
                "buffer_size": buffer_size,
                "loss": loss.item(),
            })

            if step % 100 == 0:
                print(f"[Step {step:04d}] Buffer: {buffer_size} | Reward: {reward:.4f} | Error: ")
                                
            step += 1
            

        # Curriculum step
        avg_reward = np.mean(reward_window) if reward_window else -np.inf
        if avg_reward > reward_threshold:
            buffer_size += 1
            expand_buffer()
            save_model(model, save_path)
            print(f"> Curriculum advanced. New buffer size: {buffer_size}")
            wandb.log({"curriculum_advanced": step, "avg_reward": avg_reward})
            
        

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
    train_ik_net_curriculum(vis=False, max_buffer_size=1000, save_path="checkpoints/ik_model_rl_v2_1000.pt")
