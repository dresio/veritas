import argparse
import numpy as np
import genesis as gs
import utils
import time
import tkinter as tk

# ---------------- GUI SETUP ------------------

class SphereControlGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Sphere Control")

        self.sphere_pos = [tk.DoubleVar(value=0.3) for _ in range(3)]
        self.radius = tk.DoubleVar(value=1.0)

        labels = ['X', 'Y', 'Z']
        for i in range(3):
            tk.Label(master, text=f"Sphere Pos {labels[i]}").pack()

            frame = tk.Frame(master)
            frame.pack()

            # Scale
            scale = tk.Scale(frame, from_=-2, to=2, resolution=0.01,
                             orient=tk.HORIZONTAL, variable=self.sphere_pos[i])
            scale.pack(side=tk.LEFT)

            # Entry
            entry = tk.Entry(frame, textvariable=self.sphere_pos[i], width=6)
            entry.pack(side=tk.LEFT, padx=5)

        # Radius
        tk.Label(master, text="Radius").pack()
        frame = tk.Frame(master)
        frame.pack()

        scale = tk.Scale(frame, from_=0.05, to=2.0, resolution=0.01,
                         orient=tk.HORIZONTAL, variable=self.radius)
        scale.pack(side=tk.LEFT)

        entry = tk.Entry(frame, textvariable=self.radius, width=6)
        entry.pack(side=tk.LEFT, padx=5)

    def get_sphere_pos(self):
        return tuple(var.get() for var in self.sphere_pos)

    def get_radius(self):
        return self.radius.get()


# ---------------- Sampling function ------------------
def random_point_in_sphere(radius, center):
    direction = np.random.normal(0, 1, 3)
    direction /= np.linalg.norm(direction)
    u = np.random.uniform(0, 1)
    distance = radius * (u ** (1 / 3))
    return center + distance * direction


# ---------------- Main Simulation ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    # GUI setup (must be in main thread)
    root = tk.Tk()
    gui = SphereControlGUI(root)

    gs.init(backend=gs.gpu)

    viewer_options = gs.options.ViewerOptions(
        camera_pos=(0, -3.5, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=60,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=args.vis,
    )

    plane = scene.add_entity(gs.morphs.Plane())

    links_to_keep = ["LF_FOOT"]
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/anymal_c/urdf/anymal_c.urdf",
            fixed=True,
            collision=False,
            pos=(0, 0, 1),
            links_to_keep=links_to_keep,
        ),
    )

    target_1 = scene.add_entity(
        gs.morphs.Mesh(file="meshes/axis.obj", scale=0.2),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )

    scene.build()

    joints_name = ("LF_HAA", "LF_HFE", "LF_KFE")
    dofs_idx_local = [robot.get_joint(name).dofs_idx_local[0] for name in joints_name]
    end_effector = robot.get_link("LF_FOOT")
    target_quat = np.array([1, 0, 0, 0])

    while True:
        # Keep GUI responsive
        root.update()

        # Read GUI values
        sphere_pos = gui.get_sphere_pos()
        radius = gui.get_radius()
        
        # Sample random point
        end_effector_pos = random_point_in_sphere(radius, sphere_pos)

        # Move visual target
        target_1.set_qpos(np.concatenate([end_effector_pos, target_quat]))

        # Solve IK
        qpos = robot.inverse_kinematics_multilink(
            links=[end_effector],
            poss=[end_effector_pos],
            dofs_idx_local=dofs_idx_local,
        )
        robot.set_qpos(qpos)

        scene.step()
        
        scene.clear_debug_objects()
        scene.draw_debug_sphere(pos=sphere_pos, radius=radius, color=(0.5, 0.5, 1, 0.3))

        print(f"Sphere Pos: {sphere_pos}, Radius: {radius:.2f}")

if __name__ == "__main__":
    main()
