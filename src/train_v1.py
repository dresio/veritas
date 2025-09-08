import torch
from torch import nn
from torch.optim import Adam
from ik_model import IKNet
from utils import sample_point

import genesis as gs
import os


def get_reward(pred_qpos, target_pos, robot, end_effector, dofs_idx_local):
    # Get current pose
    robot.set_qpos(pred_qpos.detach().cpu().numpy())
    robot.scene.step()
    ee_pos = robot.get_link(end_effector.name).get_pos()

    # Get how ik would solve
    ik_qpos = robot.inverse_kinematics_multilink(
            links=[end_effector],  # IK targets
            poss=[target_pos],
            dofs_idx_local=dofs_idx_local,  # IK wrt these dofs
    )
    
    # Joint angle error
    joint_error = pred_qpos - torch.tensor(ik_qpos, device=pred_qpos.device)
    reward = -torch.norm(joint_error)  # Negative distance = higher reward when close

    # End effector distance error bonus
    ee_error = target_pos - ee_pos
    task_space_bonus = -torch.norm(torch.tensor(ee_error))

    # Weighted combination 
    total_reward = reward + (0.1 * task_space_bonus)

    return total_reward.item() 


def train_ik_net_rl(vis=False, steps=10000, save_path="checkpoints/ik_model_rl.pt"):
    # Initialize Genesis
    gs.init(backend=gs.gpu)

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
        gs.morphs.URDF(file="urdf/panda_bullet/panda_nohand.urdf", fixed=True, collision=False)
    )

    scene.build()

    end_effector = panda.get_link("link7")
    joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    dofs_idx_local = [panda.get_joint(name).dofs_idx_local[0] for name in joints]

    # Policy network
    model = IKNet()
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Make policy stochastic by defining standard deviation
    action_std = 0.05
    dist_fn = torch.distributions.Normal

    # Sampling params
    inner_radius = 0.14
    outer_radius = 0.7
    sphere_pos = (0, 0, 0.33)
    cylinder_pos = (0, 0, 0.35)
    cylinder_height = 0.7

    # Training loop
    for step in range(steps):
        target_pos = sample_point(
            sphere_radius=outer_radius,
            sphere_center=sphere_pos,
            cylinder_radius=inner_radius,
            cylinder_height=cylinder_height,
            cylinder_center=cylinder_pos,
        )
        target_pos = torch.tensor(target_pos, dtype=torch.float32)

        # Get joint angles from model. The sampled distribution will be taken with these parameters as mean
        qpos_mean = model(target_pos)

        # Sample an action from a Normal distribution
        distribution = dist_fn(qpos_mean, action_std)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum()

        # Evaluate reward
        reward = get_reward(action, target_pos, panda, end_effector, dofs_idx_local)

        # REINFORCE update: maximize reward => minimize -reward * log_prob
        loss = -log_prob * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"[Step {step:04d}] Reward: {reward:.4f}")

    # Save trained policy
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nRL-trained model saved to: {save_path}")

if __name__ == "__main__":
    train_ik_net_rl(vis=False, steps=10000)