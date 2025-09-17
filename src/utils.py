import numpy as np
import torch
from dataclasses import dataclass, field

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

def sample_tensor(workspace: IKWorkspace):
    """

    Args:
        workspace (IKWorkspace, optional): _description_. IK Workspace paramters.

    Returns:
        torch.tensor: _description_
    """
    return torch.tensor(sample_point(workspace), dtype=torch.float32)

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

def compute_fk_loss(pred_qpos, target_pos, robot, end_effector, dofs_idx_local):
    """Set qpos, step sim, and compute EE error."""
    robot.set_qpos(pred_qpos.detach().cpu().numpy())
    robot.scene.step()

    ee_pos = robot.get_link(end_effector.name).get_pos()
    ee_error = torch.norm(torch.tensor(ee_pos, dtype=torch.float32) - target_pos)

    # Soft penalty: allow some tolerance
    if ee_error < 0.1:
        return ee_error
    else:
        return 0.1 + 0.01 * (ee_error - 0.1)

def compute_sampling_cost(candidate: torch.Tensor, buffer, workspace: IKWorkspace, min_dist=0.01):
    # Convert candidate to numpy
    candidate_np = candidate.numpy()

    cost = 0.0

    # Repulsion from previous points
    for point in buffer:
        dist = torch.norm(candidate - point.cpu())
        cost += torch.exp(-dist / min_dist)  # Exponential repulsion

    # Penalty for being outside spherical workspace
    sphere_dist = np.linalg.norm(candidate_np - workspace.sphere_center)
    if sphere_dist > workspace.sphere_radius:
        cost += 1e6  # Huge penalty

    # Penalty for being *inside* cylindrical keep-out zone
    xy_offset = candidate_np[:2] - workspace.cylinder_center[:2]
    radial_dist = np.linalg.norm(xy_offset)
    z = candidate_np[2]
    z_low = workspace.cylinder_center[2] - workspace.cylinder_height / 2
    z_high = workspace.cylinder_center[2] + workspace.cylinder_height / 2

    if radial_dist < workspace.cylinder_radius and z_low <= z <= z_high:
        cost += 1e6  # Also heavy penalty

    return cost

def add_point_to_buffer(workspace: IKWorkspace, buffer=None, step_size=0.5, num_candidates=20):
    if buffer is None:
        # First target - sample freely
        buffer = []
        new_target = sample_point(workspace)
    else:
        prev_target = buffer[-1].cpu()
        candidates = []
        costs = []
        
        for _ in range(num_candidates):
            # Sample a nearby point
            perturbation = (torch.randn(3) * step_size).cpu()
            candidate = prev_target + perturbation

            if check_sample_valid(workspace, candidate.numpy()):
                cost = compute_sampling_cost(candidate, buffer, workspace)
                candidates.append(candidate)
                costs.append(cost)

        if candidates:
            best_idx = torch.argmin(torch.tensor(costs))
            new_target = candidates[best_idx]
        else:
            # Fallback: sample a random valid point
            new_target = sample_point(workspace)

    buffer.append(torch.tensor(new_target, dtype=torch.float32))
    return buffer