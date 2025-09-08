import numpy as np
import torch


def sample_point(sphere_radius=3.0, sphere_center=np.array([0.0, 0.0, 0.5]), cylinder_radius=0.1, cylinder_height=1.0, cylinder_center=np.array([0.0, 0.0, 0.5])):
    """
    Samples a point inside of a sphere with a cylindrical keep-out zone in the center. Currently uses Rejection Sampling which is inefficient but simple.
    Args:
        sphere_radius (float, optional): _description_. Defaults to 3.
        sphere_center (_type_, optional): _description_. Defaults to np.array([0.0, 0.0, 0.5]).
        cylinder_radius (float, optional): _description_. Defaults to 0.1.
        cylinder_height (float, optional): _description_. Defaults to 1.0.
        cylinder_center (_type_, optional): _description_. Defaults to np.array([0.0, 0.0, 0.5]).

    Returns:
        np.array: _description_
    """
    
    
    point = sample_sphere(sphere_radius, sphere_center)
    while check_in_cylinder(cylinder_radius, cylinder_height, cylinder_center, point):
        point = sample_sphere(sphere_radius, sphere_center)
    return point


def sample_tensor(sphere_radius=3.0, sphere_center=np.array([0.0, 0.0, 0.5]), cylinder_radius=0.1, cylinder_height=1.0, cylinder_center=np.array([0.0, 0.0, 0.5])):
    """

    Args:
        sphere_radius (float, optional): _description_. Defaults to 3.
        sphere_center (np.array, optional): _description_. Defaults to np.array([0.0, 0.0, 0.5]).
        cylinder_radius (float, optional): _description_. Defaults to 0.1.
        cylinder_height (float, optional): _description_. Defaults to 1.0.
        cylinder_center (np.array, optional): _description_. Defaults to np.array([0.0, 0.0, 0.5]).

    Returns:
        torch.tensor: _description_
    """
    return torch.tensor(sample_point(sphere_radius, sphere_center, cylinder_radius, cylinder_height, cylinder_center), dtype=torch.float32)

def sample_sphere(radius = 1.0, center = np.array([0.0, 0.0, 0.0])):
    """
    Samples a point uniformly within a sphere of radius R centered at center.
    Args:
        radius (float, optional): _description_. Defaults to 1.0.
        center (np.array, optional): _description_. Defaults to np.array([0.0, 0.0, 0.0]).
    Returns:
        np.array: _description_
    """
    # Sample a random direction (unit vector) using a 3D Gaussian
    vec = np.random.normal(size=3)
    vec /= np.linalg.norm(vec)

    # Sample radius with cubic root to ensure uniform volume distribution
    random_radius = radius * np.cbrt(np.random.uniform())

    # Return the final point
    return (vec * random_radius) + center

def check_in_cylinder(radius = 1.0, height = 1.0, center = np.array([0.0, 0.0, 0.0]), point = np.array([0.0, 0.0, 0.0])):
    """
    Checks if a point is inside a cylinder of given radius, height, and center.
    Args:
        radius (float, optional): _description_. Defaults to 1.0.
        height (float, optional): _description_. Defaults to 1.0.
        center (np.array, optional): _description_. Defaults to np.array([0.0, 0.0, 0.0]).
        point (np.array, optional): _description_. Defaults to np.array([0.0, 0.0, 0.0]).
        
    Returns:
        bool: _description_
    """
    # Check height bounds
    if point[2] < center[2] - (height/2) or point[2] > center[2] + (height/2):
        return False

    # Check radial distance from the cylinder's central axis
    radial_distance = np.linalg.norm(point[:2] - center[:2])
    if radial_distance > radius:
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
