import numpy as np
import os
from scipy.spatial.transform import Rotation as R

def create_transform(vec):
    x,y,z,rx,ry,rz = vec
    transform = np.eye(4)
    transform[:3,:3] = R.from_euler('xyz', [rx,ry,rz], degrees=True).as_matrix()
    transform[:3,3] = [x,y,z]
    return transform

def transform_to_vec(transform):
    x,y,z = transform[:3,3]
    rx,ry,rz = R.from_matrix(transform[:3,:3]).as_euler('xyz', degrees=True)
    return [x,y,z,rx,ry,rz]


def euler_to_quat(r, p, y):
    sr, sp, sy = np.sin(r / 2.0), np.sin(p / 2.0), np.sin(y / 2.0)
    cr, cp, cy = np.cos(r / 2.0), np.cos(p / 2.0), np.cos(y / 2.0)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def urdf_joint_to_kdl_joint(jnt):
    import PyKDL
    kdl = PyKDL
    origin_frame = urdf_pose_to_kdl_frame(jnt.origin)
    if jnt.joint_type == "fixed":
        # SE3
        if _get_stretch_version() > 1:
            return kdl.Joint(jnt.name, kdl.Joint.Fixed)
        else:
            return kdl.Joint(jnt.name, getattr(kdl.Joint, "None"))
    axis = kdl.Vector(*jnt.axis)
    if jnt.joint_type == "revolute":
        return kdl.Joint(
            jnt.name, origin_frame.p, origin_frame.M * axis, kdl.Joint.RotAxis
        )
    if jnt.joint_type == "continuous":
        return kdl.Joint(
            jnt.name, origin_frame.p, origin_frame.M * axis, kdl.Joint.RotAxis
        )
    if jnt.joint_type == "prismatic":
        return kdl.Joint(
            jnt.name, origin_frame.p, origin_frame.M * axis, kdl.Joint.TransAxis
        )
    print("Unknown joint type: %s." % jnt.joint_type)
    return kdl.Joint(jnt.name, kdl.Joint.Fixed)


def urdf_pose_to_kdl_frame(pose):
    import PyKDL
    kdl = PyKDL
    pos = [0.0, 0.0, 0.0]
    rot = [0.0, 0.0, 0.0]
    if pose is not None:
        if pose.position is not None:
            pos = pose.position
        if pose.rotation is not None:
            rot = pose.rotation
    return kdl.Frame(kdl.Rotation.Quaternion(*euler_to_quat(*rot)), kdl.Vector(*pos))


def urdf_inertial_to_kdl_rbi(i):
    import PyKDL
    kdl = PyKDL
    origin = urdf_pose_to_kdl_frame(i.origin)
    rbi = kdl.RigidBodyInertia(
        i.mass,
        origin.p,
        kdl.RotationalInertia(
            i.inertia.ixx,
            i.inertia.iyy,
            i.inertia.izz,
            i.inertia.ixy,
            i.inertia.ixz,
            i.inertia.iyz,
        ),
    )
    return origin.M * rbi


##
# Returns a PyKDL.Tree generated from a urdf_parser_py.urdf.URDF object.
def kdl_tree_from_urdf_model(urdf):
    import PyKDL
    kdl = PyKDL
    root = urdf.get_root()
    tree = kdl.Tree(root)

    def add_children_to_tree(parent):
        if parent in urdf.child_map:
            for joint, child_name in urdf.child_map[parent]:
                child = urdf.link_map[child_name]
                if child.inertial is not None:
                    kdl_inert = urdf_inertial_to_kdl_rbi(child.inertial)
                else:
                    kdl_inert = kdl.RigidBodyInertia()
                kdl_jnt = urdf_joint_to_kdl_joint(urdf.joint_map[joint])
                kdl_origin = urdf_pose_to_kdl_frame(urdf.joint_map[joint].origin)
                kdl_sgm = kdl.Segment(child_name, kdl_jnt, kdl_origin, kdl_inert)
                tree.addSegment(kdl_sgm, parent)
                add_children_to_tree(child_name)

    add_children_to_tree(root)
    return tree


def _get_stretch_version():
    path = "/etc/hello-robot"
    files = os.listdir(path)
    fleet_id = None
    for file in files:
        if file.startswith("stretch-"):
            fleet_id = file
            break
    stretch, version, serial = fleet_id.split("-")

    return int(version[-1])
