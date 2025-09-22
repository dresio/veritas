import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Union
from tqdm import tqdm
from scipy.spatial import KDTree

@dataclass
class IKWorkspace:
    """ Parameter for IK Linkage workspace area. Allows for spherical range and cylindrical keep-out zone."""
    sphere_radius: float = 3.0
    sphere_center: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.5]))
    cylinder_radius: float = 0.1
    cylinder_height: float = 1.0
    cylinder_center: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.5]))
    
def sample_point(workspace: IKWorkspace):
    """
    Samples a point inside of a sphere with a cylindrical keep-out zone in the center. Currently uses Rejection Sampling which is inefficient but simple.
    Args:
        workspace (IKWorkspace, optional): _description_. IK Workspace paramters.

    Returns:
        np.array: _description_
    """
    
    point = sample_sphere(workspace)
    while check_in_cylinder(workspace, point):
        point = sample_sphere(workspace)
    return point

def check_sample_valid(workspace: IKWorkspace, point=np.array([0.0,0.0,0.0])) -> bool:
    """
    Checks if a point is within the valid sampling region (inside sphere but outside cylinder).
    Args:
        point (np.array, optional): _description_. Defaults to np.array([0.0,0.0,0.0]).
        sphere_radius (float, optional): _description_. Defaults to 3.
        sphere_center (np.array, optional): _description_. Defaults to np.array([0.0, 0.0, 0.5]).
        cylinder_radius (float, optional): _description_. Defaults to 0.1.
        cylinder_height (float, optional): _description_. Defaults to 1.0.
        cylinder_center (np.array, optional): _description_. Defaults to np.array([0.0, 0.0, 0.5]).

    Returns:
        bool: _description_
    """
    # Check if point is inside the sphere
    sphere_dist = np.linalg.norm(point - workspace.sphere_center)
    inside_sphere = sphere_dist <= workspace.sphere_radius
    
    # Check if point is inside the cylinder
    inside_cylinder = check_in_cylinder(
        workspace=workspace,
        point=point
    )

    # Valid if inside the sphere AND outside the cylinder
    return inside_sphere and not inside_cylinder

def sample_sphere(workspace: IKWorkspace):
    """
    Samples a point uniformly within spherical workspace
    Args:
        workspace (IKWorkspace, optional): _description_. IK Workspace paramters.
    Returns:
        np.array: _description_
    """
    # Sample a random direction (unit vector) using a 3D Gaussian
    vec = np.random.normal(size=3)
    vec /= np.linalg.norm(vec)

    # Sample radius with cubic root to ensure uniform volume distribution
    random_radius = workspace.sphere_radius * np.cbrt(np.random.uniform())

    # Return the final point
    return (vec * random_radius) + workspace.sphere_center

def check_in_cylinder(workspace: IKWorkspace, point = np.array([0.0, 0.0, 0.0])):
    """
    Checks if a point is inside a cylinder of given radius, height, and center.
    Args:
        workspace (IKWorkspace, optional): _description_. IK Workspace paramters.
        point (np.array, optional): _description_. Defaults to np.array([0.0, 0.0, 0.0]).
        
    Returns:
        bool: _description_
    """
    # Check height bounds
    if point[2] < workspace.cylinder_center[2] - (workspace.cylinder_height/2) or point[2] > workspace.cylinder_center[2] + (workspace.cylinder_height/2):
        return False

    # Check radial distance from the cylinder's central axis
    radial_distance = np.linalg.norm(point[:2] - workspace.cylinder_center[:2])
    if radial_distance > workspace.cylinder_radius:
        return False

    return True

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

        
    for _ in tqdm(range(buffer_size - 1), desc="Generating curriculum buffer"):
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

        buffer.append(torch.tensor(new_target, dtype=torch.float32, device=device))

    return torch.stack(buffer).to(device)