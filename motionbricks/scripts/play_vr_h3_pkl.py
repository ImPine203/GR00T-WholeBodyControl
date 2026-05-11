"""Inspect and play one retargeted VR H3 pickle clip in MuJoCo."""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from motionbricks.helper.mujoco_helper import get_mujoco_converter
from motionbricks.motionlib.core.utils.rotations import quaternion_to_matrix
from prepare_vr_h3_data import (
    EXPECTED_FULL_DIM,
    EXPECTED_GLOBAL_DIM,
    clip_to_local_rotations,
    convert_clip,
    load_vr_h3_motion_rep,
    load_vr_h3_pickle,
)


VR_H3_DOF_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
]
CSV_DEFAULT_FPS = 120


def select_clip(loaded, clip_name):
    if not isinstance(loaded, dict):
        raise ValueError(f"Top-level object must be dict, got {type(loaded).__name__}")

    clip_items = [(name, clip) for name, clip in loaded.items() if isinstance(clip, dict)]
    if not clip_items:
        raise ValueError("No dict clip entries found in pkl")

    if clip_name is None:
        return clip_items[0]

    for name, clip in clip_items:
        if str(name) == clip_name:
            return name, clip

    available = ", ".join(str(name) for name, _ in clip_items[:20])
    raise ValueError(f"Clip '{clip_name}' not found. Available clips: {available}")


def load_vr_h3_csv(path, root_euler_order):
    data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64))
    if data.dtype.names is None:
        raise ValueError(f"CSV '{path}' does not have a header")

    missing = [
        name for name in [
            "root_translateX", "root_translateY", "root_translateZ",
            "root_rotateX", "root_rotateY", "root_rotateZ",
            *[joint_name + "_dof" for joint_name in VR_H3_DOF_JOINT_NAMES],
        ] if name not in data.dtype.names
    ]
    if missing:
        raise ValueError(f"CSV '{path}' is missing columns: {missing}")

    root_trans = np.stack([
        data["root_translateX"],
        data["root_translateY"],
        data["root_translateZ"],
    ], axis=-1).astype(np.float32) / 100.0
    root_euler_deg = np.stack([
        data["root_rotateX"],
        data["root_rotateY"],
        data["root_rotateZ"],
    ], axis=-1)
    root_rot_xyzw = Rotation.from_euler(root_euler_order, root_euler_deg, degrees=True).as_quat().astype(np.float32)
    dof_deg = np.stack([data[joint_name + "_dof"] for joint_name in VR_H3_DOF_JOINT_NAMES], axis=-1)

    return {
        "Frame": data["Frame"].astype(np.int64) if "Frame" in data.dtype.names else np.arange(data.shape[0]),
        "fps": CSV_DEFAULT_FPS,
        "root_trans_offset": root_trans,
        "root_rot": root_rot_xyzw,
        "dof": np.deg2rad(dof_deg).astype(np.float32),
    }


def load_input_clip(path, clip_name, csv_root_euler_order):
    if path.suffix.lower() == ".csv":
        return path.stem, load_vr_h3_csv(path, csv_root_euler_order)

    loaded = load_vr_h3_pickle(path)
    return select_clip(loaded, clip_name)


def describe_array(name, value):
    arr = np.asarray(value)
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
    if arr.size and np.issubdtype(arr.dtype, np.number):
        finite = np.isfinite(arr)
        finite_count = int(finite.sum())
        print(
            f"  finite={finite_count}/{arr.size}, "
            f"min={np.nanmin(arr):.6f}, max={np.nanmax(arr):.6f}, "
            f"mean={np.nanmean(arr):.6f}"
        )


def print_clip_summary(path, clip_name, clip, full_motion, global_motion, qpos, model, source, root_rotation):
    print(f"Input: {path}")
    print(f"Clip: {clip_name}")
    print(f"Source: {source}")
    if root_rotation is not None:
        print(f"Root rotation source: {root_rotation}")
    print(f"Keys: {sorted(clip.keys())}")
    for key in ["fps", "root_trans_offset", "root_rot", "pose_aa", "dof", "smpl_joints"]:
        if key in clip:
            describe_array(key, clip[key])
    if full_motion is not None:
        print(f"full_motion: shape={tuple(full_motion.shape)}")
    if global_motion is not None:
        print(f"global_motion: shape={tuple(global_motion.shape)}")
    print(f"mujoco_qpos: shape={tuple(qpos.shape)}")
    print(f"mujoco_model: nq={model.nq}, nv={model.nv}, njnt={model.njnt}")


def convert_clip_with_root_rot(clip, motion_rep, quat_order):
    if "root_trans_offset" not in clip:
        raise ValueError("missing root_trans_offset")
    if "root_rot" not in clip:
        raise ValueError("missing root_rot")

    translation = torch.from_numpy(np.asarray(clip["root_trans_offset"])).float()
    local_rots = clip_to_local_rotations(clip, motion_rep.skeleton.nbjoints)
    root_quat = torch.from_numpy(np.asarray(clip["root_rot"])).float()
    if quat_order == "xyzw":
        root_quat = root_quat[:, [3, 0, 1, 2]]
    local_rots[:, 0] = quaternion_to_matrix(root_quat)

    if local_rots.shape[0] != translation.shape[0]:
        raise ValueError("pose_aa and root_trans_offset frame counts do not match")

    full_motion = motion_rep.dual_rep(
        {"local_joint_rots": local_rots, "translation": translation},
        to_normalize=False,
    )
    global_motion = motion_rep.dual_rep.get_feature_subset(full_motion, mode="global")

    if full_motion.shape[-1] != EXPECTED_FULL_DIM:
        raise ValueError(f"expected full dim {EXPECTED_FULL_DIM}, got {full_motion.shape[-1]}")
    if global_motion.shape[-1] != EXPECTED_GLOBAL_DIM:
        raise ValueError(f"expected global dim {EXPECTED_GLOBAL_DIM}, got {global_motion.shape[-1]}")
    if not torch.isfinite(full_motion).all():
        raise ValueError("full motion contains NaN or Inf")

    return full_motion, global_motion


def convert_to_qpos(clip, motion_rep, skeleton_xml, root_rotation, root_quat_order):
    if root_rotation == "pose_aa":
        full_motion, global_motion = convert_clip(clip, motion_rep)
    else:
        full_motion, global_motion = convert_clip_with_root_rot(clip, motion_rep, root_quat_order)

    converter = get_mujoco_converter(motion_rep, skeleton_xml)
    qpos = converter.convert_motion_features_to_mujoco_qpos(
        global_motion[None], motion_rep, is_normalized=False
    )[0]

    root_quat = qpos[:, 3:7].clone()
    qpos[:, 3:7] = root_quat[:, [3, 0, 1, 2]]
    return full_motion, global_motion, qpos


def root_quat_to_wxyz(root_rot, quat_order):
    root_quat = torch.from_numpy(np.asarray(root_rot)).float()
    if quat_order == "xyzw":
        root_quat = root_quat[:, [3, 0, 1, 2]]
    return root_quat


def convert_dof_to_qpos(clip, model, root_quat_order):
    if "root_trans_offset" not in clip:
        raise ValueError("missing root_trans_offset")
    if "root_rot" not in clip:
        raise ValueError("missing root_rot")
    if "dof" not in clip:
        raise ValueError("missing dof")

    root_trans = torch.from_numpy(np.asarray(clip["root_trans_offset"])).float()
    dof = torch.from_numpy(np.asarray(clip["dof"])).float()
    if dof.ndim != 2 or dof.shape[1] != len(VR_H3_DOF_JOINT_NAMES):
        raise ValueError(f"dof must have shape [T, {len(VR_H3_DOF_JOINT_NAMES)}], got {tuple(dof.shape)}")
    if root_trans.shape[0] != dof.shape[0]:
        raise ValueError("root_trans_offset and dof frame counts do not match")

    qpos = torch.zeros((dof.shape[0], model.nq), dtype=torch.float32)
    qpos[:, :3] = root_trans
    qpos[:, 3:7] = root_quat_to_wxyz(clip["root_rot"], root_quat_order)

    for dof_idx, joint_name in enumerate(VR_H3_DOF_JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint '{joint_name}' not found in MuJoCo model")
        qpos_id = model.jnt_qposadr[joint_id]
        qpos[:, qpos_id] = dof[:, dof_idx]

    return qpos


def print_joint_limit_summary(model, qpos):
    violations = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if not model.jnt_limited[joint_id]:
            continue
        qpos_id = model.jnt_qposadr[joint_id]
        values = qpos[:, qpos_id].numpy()
        low, high = model.jnt_range[joint_id]
        below = float(np.maximum(low - values, 0.0).max())
        above = float(np.maximum(values - high, 0.0).max())
        over = max(below, above)
        if over > 1e-5:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            violations.append((over, name, float(values.min()), float(values.max()), float(low), float(high)))

    if not violations:
        print("joint_limit_violations: none")
        return

    violations.sort(reverse=True)
    print(f"joint_limit_violations: {len(violations)} joints")
    for over, name, min_value, max_value, low, high in violations[:10]:
        print(
            f"  {name}: max_over={over:.6f}, "
            f"value_range=[{min_value:.6f}, {max_value:.6f}], "
            f"limit=[{low:.6f}, {high:.6f}]"
        )


def play_qpos(model, qpos, fps, speed, loop, no_viewer):
    data = mujoco.MjData(model)
    frame_dt = 1.0 / fps / speed

    if no_viewer:
        for frame in range(qpos.shape[0]):
            data.qpos[:] = qpos[frame].numpy()
            mujoco.mj_forward(model, data)
        print(f"Headless playback completed: {qpos.shape[0]} frames")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame = 0
        while viewer.is_running():
            start = time.time()
            data.qpos[:] = qpos[frame].numpy()
            mujoco.mj_forward(model, data)
            viewer.cam.lookat[:] = data.qpos[:3]
            viewer.sync()

            frame += 1
            if frame >= qpos.shape[0]:
                if not loop:
                    break
                frame = 0

            sleep_time = frame_dt - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser(
        description="Load one VR H3 retargeted pkl/csv and play it on the VR H3 MuJoCo robot."
    )
    parser.add_argument("input", type=str, help="Path to a retargeted .pkl or .csv file")
    parser.add_argument("--clip", type=str, default=None, help="Clip key inside a pkl. Defaults to first clip.")
    parser.add_argument("--config", type=str, default="out/motionbricks_vr_h3_vqvae/version_1/hparams.yaml")
    parser.add_argument("--skeleton_xml", type=str,
                        default="assets/skeletons/vr_h3/mjcf/origin/vr_h3_1_with_rh56e2_hand.xml")
    parser.add_argument("--scene_xml", type=str,
                        default="assets/skeletons/vr_h3/mjcf/origin/vr_h3_1_with_rh56e2_hand_scene.xml")
    parser.add_argument("--source", choices=["dof", "motion_feature"], default="dof",
                        help="dof plays the retargeted robot joint angles directly; motion_feature tests the training conversion.")
    parser.add_argument("--root_rotation", choices=["pose_aa", "root_rot"], default="pose_aa",
                        help="Root rotation source for --source motion_feature.")
    parser.add_argument("--root_quat_order", choices=["wxyz", "xyzw"], default="xyzw",
                        help="Quaternion order used for root_rot.")
    parser.add_argument("--csv_root_euler_order", type=str, default="xyz",
                        help="Euler order for CSV root_rotateX/Y/Z columns.")
    parser.add_argument("--fps", type=float, default=None, help="Override playback FPS. Defaults to input fps.")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--no_viewer", action="store_true")
    args = parser.parse_args()

    path = Path(args.input)
    clip_name, clip = load_input_clip(path, args.clip, args.csv_root_euler_order)

    model = mujoco.MjModel.from_xml_path(args.scene_xml)
    if args.source == "dof":
        full_motion = None
        global_motion = None
        qpos = convert_dof_to_qpos(clip, model, args.root_quat_order)
    else:
        motion_rep = load_vr_h3_motion_rep(args.config)
        full_motion, global_motion, qpos = convert_to_qpos(
            clip, motion_rep, args.skeleton_xml, args.root_rotation, args.root_quat_order
        )

    if qpos.shape[-1] != model.nq:
        raise ValueError(f"Converted qpos dim {qpos.shape[-1]} does not match MuJoCo nq {model.nq}")

    fps = float(args.fps or clip.get("fps", 30))
    print_clip_summary(
        path, clip_name, clip, full_motion, global_motion, qpos, model,
        args.source, args.root_rotation if args.source == "motion_feature" else None,
    )
    print_joint_limit_summary(model, qpos.cpu())
    play_qpos(model, qpos.cpu(), fps, args.speed, args.loop, args.no_viewer)


if __name__ == "__main__":
    main()
