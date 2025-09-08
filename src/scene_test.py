import argparse
import numpy as np
import genesis as gs
import utils
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.gpu)

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(0, -3.5, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=60,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    panda = scene.add_entity(
        gs.morphs.URDF(file="urdf/panda_bullet/panda_nohand.urdf", fixed=True, collision=False),
    )

    target_1 = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.2,
        ),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )

    inner_radius = 0.14
    outer_radius = 0.7
    sphere_pos = (0, 0, 0.33)
    cylinder_pos = (0, 0, 0.35)
    cylinder_height = 0.7
    sphere = scene.add_entity(
        gs.morphs.Sphere(radius=outer_radius, pos=sphere_pos, fixed=True, collision=False),
        surface=gs.surfaces.Default(color=(0.5, 0.5, 1), opacity=0.3),
    )

    keep_out_cylinder = scene.add_entity(
        gs.morphs.Cylinder(
            radius=inner_radius,
            height=cylinder_height,
            pos=cylinder_pos,
            fixed=True,
            collision=False,
        ),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5), opacity=0.3),
    )
    ########################## build ##########################
    scene.build()

    joints_name = (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    )

    dofs_idx_local = [panda.get_joint(name).dofs_idx_local[0] for name in joints_name]
    target_quat = np.array([1, 0, 0, 0])
    end_effector = panda.get_link("link7")

    while True:
        end_effector_pos = utils.sample_point(
            sphere_radius=outer_radius,
            sphere_center=sphere_pos,
            cylinder_radius=inner_radius,
            cylinder_height=cylinder_height,
            cylinder_center=cylinder_pos,
        )

        target_1.set_qpos(np.concatenate([end_effector_pos, target_quat]))

        qpos = panda.inverse_kinematics_multilink(
            links=[end_effector],  # IK targets
            poss=[end_effector_pos],
            dofs_idx_local=dofs_idx_local,  # IK wrt these dofs
        )

        panda.set_qpos(qpos)
        scene.step()


if __name__ == "__main__":
    main()