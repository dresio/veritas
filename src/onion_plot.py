import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import pickle
from scipy.interpolate import griddata

from ik_model import IKNet
import genesis as gs

def sample_sphere(radius: float, center: np.array, num_points: int):
    """Samples sphere in a regular interval pattern (Fibonacci Lattice)

    Args:
        radius (float): radius to sample over
        center (np.array): center of the sphere
        num_points (int): number of points to sample

    Returns:
        _type_: array of sampled points
    """
    points = []
    offset = 2.0 / num_points
    increment = math.pi * (3.0 - math.sqrt(5.0))

    for i in range(num_points):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - y * y)
        phi = i * increment
        x = math.cos(phi) * r
        z = math.sin(phi) * r
        point = np.array([x, y, z]) * radius + np.array(center)
        points.append(point)

    return np.array(points)

def get_positional_error(model, target_pos, robot, end_effector):
    """Gets positional error from models prediction

    Args:
        model (_type_): pytorch model of ik solver
        target_pos (_type_): position of target to reach
        robot (_type_): genesis robot model
        end_effector (_type_): link to end effector from robot

    Returns:
        float: positional error (L2 norm)
    """
    target_tensor = torch.tensor(target_pos, dtype=torch.float32)
    pred_qpos = model(target_tensor)
    robot.set_qpos(pred_qpos.detach().cpu().numpy())
    robot.scene.step()
    ee_pos = robot.get_link(end_effector.name).get_pos()
    error = np.linalg.norm(np.array(target_pos) - ee_pos.cpu().numpy())
    return error

def plot_hemisphere_2d(points, errors, buffer_points, save_path, title):
    """Plots a hemisphere in 2D with a heatmap of errors and overlays buffer points

    Args:
        points (_type_): list of locations for each of the points error was calculated for
        errors (_type_): errors at each value of points
        buffer_points (_type_): 3D location of where the model was trained on
        save_path (_type_): save path for the plot
        title (_type_): title of the plot
    """
    errors = np.array(errors)
    norm_errors = (errors - errors.min()) / (np.ptp(errors) + 1e-8)

    # Project 3D to 2D (top-down)
    xy = points[:, [0, 1]]  # Only X and Y

    # Create a grid over 2D space
    grid_res = 500
    grid_x, grid_y = np.mgrid[
        xy[:, 0].min():xy[:, 0].max():complex(grid_res),
        xy[:, 1].min():xy[:, 1].max():complex(grid_res)
    ]

    # Interpolate errors over the grid
    grid_z = griddata(xy, norm_errors, (grid_x, grid_y), method='linear')

    # Mask values outside circle
    radius = np.max(np.linalg.norm(xy, axis=1))
    mask = np.sqrt(grid_x**2 + grid_y**2) > radius
    grid_z[mask] = np.nan

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 6))
    c = ax.imshow(
        grid_z.T,
        extent=(xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max()),
        origin='lower',
        cmap='coolwarm',
        interpolation='bilinear'
    )
    plt.colorbar(c, ax=ax, label="Normalized Positional Error")

    # Plot buffer points
    if buffer_points is not None and len(buffer_points) > 0:
        buffer_points = np.array(buffer_points)
        ax.scatter(buffer_points[:, 0], buffer_points[:, 1], color='limegreen', s=20, label='Buffer Points')

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")

def run_onion_shell_analysis_flat(radius, model, robot, end_effector, dofs_idx_local,
                                  buffer_points=None, center=(0,0,0.33), num_points=300, save_dir="output_plots"):

    os.makedirs(save_dir, exist_ok=True)
    points = sample_sphere(radius=radius, center=center, num_points=num_points)

    upper_pts, upper_errs = [], []
    lower_pts, lower_errs = [], []

    for point in points:
        error = get_positional_error(model, point, robot, end_effector)
        if point[2] >= center[2]:
            upper_pts.append(point)
            upper_errs.append(error)
        else:
            lower_pts.append(point)
            lower_errs.append(error)

    # Split buffer points
    upper_buffer, lower_buffer = [], []
    if buffer_points is not None:
        for bp in buffer_points:
            if bp[2] >= center[2]:
                upper_buffer.append(bp)
            else:
                lower_buffer.append(bp)

    plot_hemisphere_2d(
        points=np.array(upper_pts),
        errors=upper_errs,
        buffer_points=upper_buffer,
        save_path=os.path.join(save_dir, f"radius_{radius:.2f}_upper.png"),
        title=f"Upper Hemisphere @ r={radius:.2f}"
    )

    plot_hemisphere_2d(
        points=np.array(lower_pts),
        errors=lower_errs,
        buffer_points=lower_buffer,
        save_path=os.path.join(save_dir, f"radius_{radius:.2f}_lower.png"),
        title=f"Lower Hemisphere @ r={radius:.2f}"
    )

def load_buffer(path="checkpoints/buffer.pkl"):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
if __name__ == "__main__":
    # Set up
    buffer_path = "checkpoints/buffer.pkl"
    model_path = "checkpoints/ik_model.pt"
    save_dir = "output_plots"
    center = np.array([0, 0, 0.33])
    radii = np.linspace(0.1, 0.7, num=7)
    num_points_per_shell = 300

    # Init Genesis
    gs.init(backend=gs.gpu, logging_level='warning')
    scene = gs.Scene()
    scene.add_entity(gs.morphs.Plane())
    panda = scene.add_entity(gs.morphs.URDF(file="urdf/panda_bullet/panda_nohand.urdf", fixed=True))
    scene.build()
    end_effector = panda.get_link("link7")
    joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    dofs_idx_local = [panda.get_joint(name).dofs_idx_local[0] for name in joints]

    # Load model
    model = IKNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load buffer and group by radius
    buffer = load_buffer(buffer_path)
    buffer_array = [b.cpu().numpy() if torch.is_tensor(b) else b for b in buffer]

    # Assign buffer points to their closest radius
    buffer_by_radius = {r: [] for r in radii}
    for point in buffer_array:
        dist = np.linalg.norm(point - center)
        closest_r = radii[np.argmin(np.abs(radii - dist))]
        buffer_by_radius[closest_r].append(point)

    # Run onion plots with grouped buffer points
    for radius in radii:
        print(f"> Plotting shell at radius = {radius:.2f} with {len(buffer_by_radius[radius])} buffer points")
        run_onion_shell_analysis_flat(
            radius=radius,
            model=model,
            robot=panda,
            end_effector=end_effector,
            dofs_idx_local=dofs_idx_local,
            buffer_points=buffer_by_radius[radius],
            center=center,
            num_points=num_points_per_shell,
            save_dir=save_dir
        )