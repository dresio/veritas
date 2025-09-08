import torch
import time
import numpy as np

from ik_model import IKNet
from utils import sample_point

import genesis as gs


def get_end_effector_error(qpos, panda, end_effector, target_pos):
    panda.set_qpos(qpos)
    panda.scene.step()

    # Ensure end-effector position is on CPU before using numpy
    ee_pos = panda.get_link(end_effector.name).get_pos().detach().cpu().numpy()

    # Convert target to CPU numpy
    target_np = target_pos.detach().cpu().numpy()

    error = np.linalg.norm(ee_pos - target_np)
    return error

def test_model_vs_ik(model_path="checkpoints/ik_model_rl_v2_1000.pt", trials=100):
    # Init Genesis
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
    )

    scene.add_entity(gs.morphs.Plane())
    panda = scene.add_entity(
        gs.morphs.URDF(file="urdf/panda_bullet/panda_nohand.urdf", fixed=True)
    )
    scene.build()

    end_effector = panda.get_link("link7")
    joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    dofs_idx_local = [panda.get_joint(name).dofs_idx_local[0] for name in joints]

    # Load model
    model = IKNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    model_errors, model_times = [], []
    ik_errors, ik_times = [], []

    # Sampling params
    inner_radius = 0.14
    outer_radius = 0.7
    sphere_pos = (0, 0, 0.33)
    cylinder_pos = (0, 0, 0.35)
    cylinder_height = 0.7

    for i in range(trials):
        target_pos = sample_point(
            sphere_radius=outer_radius,
            sphere_center=sphere_pos,
            cylinder_radius=inner_radius,
            cylinder_height=cylinder_height,
            cylinder_center=cylinder_pos,
        )
        target_pos = torch.tensor(target_pos, dtype=torch.float32)

        # Model testing
        start = time.time()
        with torch.no_grad():
            pred_qpos = model(target_pos)
        model_time = time.time() - start

        model_error = get_end_effector_error(
            pred_qpos.detach().cpu().numpy(), panda, end_effector, target_pos
        )

        model_errors.append(model_error)
        model_times.append(model_time)

        # IK testing
        start = time.time()
        ik_qpos = panda.inverse_kinematics_multilink(
            links=[end_effector],
            poss=[target_pos.detach().cpu().numpy()],
            dofs_idx_local=dofs_idx_local,
        )
        ik_time = time.time() - start

        ik_error = get_end_effector_error(ik_qpos, panda, end_effector, target_pos)

        ik_errors.append(ik_error)
        ik_times.append(ik_time)

        print(f"[{i+1:03d}/{trials}] Model Err: {model_error:.4f}, IK Err: {ik_error:.4f}")

    # Report results
    print("\n===== Evaluation Results =====")
    print("MODEL")
    print(f"  Mean Time   : {np.mean(model_times)*1e3:.2f} ms")
    print(f"  Mean Error  : {np.mean(model_errors):.4f}")
    print(f"  Std Dev Err : {np.std(model_errors):.4f}")

    print("\nGENESIS IK")
    print(f"  Mean Time   : {np.mean(ik_times)*1e3:.2f} ms")
    print(f"  Mean Error  : {np.mean(ik_errors):.4f}")
    print(f"  Std Dev Err : {np.std(ik_errors):.4f}")

if __name__ == "__main__":
    test_model_vs_ik(trials=100)