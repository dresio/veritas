
from utils import IKWorkspace, sample_point, check_sample_valid
import genesis as gs
import torch
import pickle
from tqdm import tqdm
from scipy.spatial import KDTree
from typing import Union
import numpy as np  


def generate_buffer(workspace: IKWorkspace, step_size: float = 0.5, num_candidates: int = 20, buffer_size: int = 1000, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    """
    Generate a curriculum buffer of points within the workspace using cost-based sampling. Now uses KDTree for efficient neighbor searches to prevent unbounded processing time.

    Args:
        workspace (IKWorkspace): Workspace specification.
        step_size (float): Distance to perturb points.
        num_candidates (int): Number of candidates to sample around previous point (more candidates mean more accurate density sampling).
        buffer_size (int): Number of target points to include in buffer.
        device (str or torch.device): Device to place final buffer on.

    Returns:
        torch.Tensor: Buffer of shape [buffer_size, 3] on the specified device.
    """
    
    buffer = []
    new_target = sample_point(workspace)
    buffer.append(torch.tensor(new_target, dtype=torch.float32))

        
    for _ in tqdm(range(buffer_size - 1), desc="Generating target points"):
        prev_target = buffer[-1].cpu()
        candidates = []
        candidate_costs = []
        
        # Build KDTree from the current buffer
        buffer_array = torch.stack(buffer).cpu().numpy()
        kdtree = KDTree(buffer_array)

        for _ in range(num_candidates):
            perturbation = torch.randn(3, device=prev_target.device) * step_size
            candidate = prev_target + perturbation

            if check_sample_valid(workspace, candidate.numpy()):
                # Use KDTree to get distance to k nearest neighbors
                k = min(5, len(buffer))  # Avoid requesting more neighbors than exist
                dists, _ = kdtree.query(candidate.numpy(), k=k)
                dists = np.atleast_1d(dists)

                cost = dists.mean() 

                candidates.append(candidate)
                candidate_costs.append(cost)

        if candidates:
            # Select candidate with lowest cost
            best_idx = torch.argmin(torch.tensor(candidate_costs))
            new_target = candidates[best_idx]
        else:
            # Fallback to a random sample if all candidates given did not pass check_sample_valid
            new_target = torch.tensor(sample_point(workspace), dtype=torch.float32)

        buffer.append(new_target.detach().clone().to(torch.float32))

    return torch.stack(buffer).to(device)

def generate_dataset():
    workspace_path="ik_workspace.json"
    workspace = None
    try:
        workspace = IKWorkspace.load_from_file(workspace_path)
        print(f"Loaded workspace from {workspace_path}")
    except Exception as e:
        print(f"Failed to load workspace from {workspace_path}, using default. Error: {e}")

    buffer_size = 10000

    buffer = generate_buffer(workspace, step_size=1, buffer_size=buffer_size)  # [B, 3]

    # Initialize Genesis
    gs.init(backend=gs.gpu, 
            logging_level='warning',
            # performance_mode=True,
            )

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
    joints_name = ("LF_HAA", "LF_HFE", "LF_KFE")
    dofs_idx_local = [robot.get_joint(name).dofs_idx_local[0] for name in joints_name]
    end_effector = robot.get_link("LF_FOOT")
    scene.build(n_envs=buffer_size, env_spacing=(1.0, 1.0))

    ik_qpos = robot.inverse_kinematics_multilink(
        links=[end_effector],
        poss=buffer.cpu().numpy().reshape(1, -1, 3),  # [1, B, 3]
        dofs_idx_local=dofs_idx_local,
        envs_idx=list(range(buffer.shape[0]))
    )  # Returns [B, N] numpy


    ik_qpos = ik_qpos[:, dofs_idx_local]  # shape: [B, len(dofs_idx_local)]

    # ik_qpos = torch.tensor(ik_qpos, dtype=torch.float32, device="cuda")  # Convert to tensor

    dataset = {
        "target_pos": buffer.cpu().numpy(),   # [B, 3]
        "joint_angles": ik_qpos.cpu().numpy()  # [B, N]
    }

    save_dataset(dataset)

def load_dataset(filename: str = "ik_dataset.pkl")-> dict:
    """_summary_

    Args:
        filename (str, optional): Filename to load dataset from. Defaults to "ik_dataset.pkl".

    Returns:
        dict: dict with "target_pos" and "joint_angles" keys.
    """
    with open(filename, "rb") as f:
        dataset = pickle.load(f)
    return dataset

def save_dataset(dataset):
    with open("ik_dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    generate_dataset()