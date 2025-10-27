import torch
import time
import numpy as np

from ik_model import IKNet
from utils import sample_point, IKWorkspace

import genesis as gs

from tqdm import tqdm

def get_end_effector_error(qpos, robot, end_effector, target_pos, dofs_idx_local=None):
    
    robot.set_dofs_position(qpos, dofs_idx_local=dofs_idx_local, zero_velocity=True)
    
    
    robot.scene.step()

    # Ensure end-effector position is on CPU before using numpy
    ee_pos = robot.get_link(end_effector.name).get_pos().detach().cpu().numpy()

    # Convert target to CPU numpy
    target_np = target_pos.detach().cpu().numpy()

    error = np.linalg.norm(ee_pos - target_np)
    return error

def test_model_vs_ik(model_path="checkpoints/ik_model.pt",  trials=1000, workspace_path="ik_workspace.json"):
    # Init Genesis
    gs.init(backend=gs.gpu, logging_level='warning',)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
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
    scene.build()

    joints_name = ("LF_HAA", "LF_HFE", "LF_KFE")
    dofs_idx_local = [robot.get_joint(name).dofs_idx_local[0] for name in joints_name]
    print(f"Controlling joints: {joints_name} at DOF indices {dofs_idx_local}, total idx length: {len(dofs_idx_local)}")
    end_effector = robot.get_link("LF_FOOT")

    # Load model
    pre_trained_model = IKNet(output_dim=len(dofs_idx_local))
    pre_trained_model.load_state_dict(torch.load(model_path))
    pre_trained_model.eval()
    
    model_errors, model_times = [], []
    ik_errors, ik_times = [], []
    model_errors_post = []

    # Generate workspace
    workspace = None
    try:
        workspace = IKWorkspace.load_from_file(workspace_path)
        print(f"Loaded workspace from {workspace_path}")
    except Exception as e:
        print(f"Failed to load workspace from {workspace_path}, using default. Error: {e}")
    
    for i in tqdm(range(trials)):
        target_pos = sample_point(workspace)
        target_pos = torch.tensor(target_pos, dtype=torch.float32)

        # Model testing
        start = time.time()
        with torch.no_grad():
            pred_qpos = pre_trained_model(target_pos)
        model_time = (time.time() - start) 

        model_error = get_end_effector_error(
            pred_qpos.detach().cpu().numpy(), robot, end_effector, target_pos, dofs_idx_local=dofs_idx_local
        )
        

        model_errors.append(model_error)
        model_times.append(model_time)

        # IK testing
        start = time.time()
        ik_qpos = robot.inverse_kinematics_multilink(
            links=[end_effector],
            poss=[target_pos.detach().cpu().numpy()],
            dofs_idx_local=dofs_idx_local,
        )
        ik_time = time.time() - start
        
        ik_qpos_np = ik_qpos.detach().cpu().numpy()[dofs_idx_local] #only get the indeces we care about
        
        ik_error = get_end_effector_error(ik_qpos_np, robot, end_effector, target_pos, dofs_idx_local=dofs_idx_local)

        ik_errors.append(ik_error)
        ik_times.append(ik_time)

        # print(f"[{i+1:03d}/{trials}] Model Err: {model_error:.4f}, IK Err: {ik_error:.4f}")

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
    print("Model vs Genesis IK")
    test_model_vs_ik(trials=5000)