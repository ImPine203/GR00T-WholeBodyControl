"""Convert retargeted VR H3 CSV clips into MotionBricks training features."""

import argparse
import io
import os
import pickle
import re
import zlib
from pathlib import Path

import joblib
import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from scipy.spatial.transform import Rotation

from motionbricks.helper.mujoco_helper import get_mujoco_converter
from motionbricks.helper.pl_util import load_motion_rep
from motionbricks.motionlib.core.utils.rotations import exp_map_to_matrix


EXPECTED_GLOBAL_DIM = 402
EXPECTED_FULL_DIM = 406
CSV_DEFAULT_FPS = 120
ACTION_SUFFIX_PATTERNS = [
    re.compile(r"_\d+__A\d+(?:_M)?$"),
    re.compile(r"_\d+_A\d+(?:_M)?$"),
    re.compile(r"__A\d+(?:_M)?$"),
]

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


def load_vr_h3_motion_rep(config_path: str):
    conf = OmegaConf.load(config_path)
    version_dir = os.path.dirname(config_path)
    with open_dict(conf):
        conf.skeleton.folder = os.path.join(version_dir, "skeleton")
        conf.motion_rep.stats.folder = os.path.join(version_dir, "stats", "motion")
    return load_motion_rep(conf)


def load_vr_h3_pickle(path: Path):
    raw = path.read_bytes()
    try:
        raw = zlib.decompress(raw)
    except zlib.error:
        pass

    try:
        return joblib.load(io.BytesIO(raw))
    except Exception:
        return pickle.loads(raw)


def clip_to_local_rotations(clip: dict, nbjoints: int):
    if "pose_aa" not in clip:
        raise ValueError("missing pose_aa")

    pose_aa = torch.from_numpy(np.asarray(clip["pose_aa"])).float()
    if pose_aa.ndim != 3 or pose_aa.shape[-1] != 3:
        raise ValueError(f"pose_aa must have shape [T, J, 3], got {tuple(pose_aa.shape)}")
    if pose_aa.shape[1] > nbjoints:
        raise ValueError(f"pose_aa has {pose_aa.shape[1]} joints, but skeleton has {nbjoints}")

    local_rots = torch.eye(3).repeat(pose_aa.shape[0], nbjoints, 1, 1)
    local_rots[:, : pose_aa.shape[1]] = exp_map_to_matrix(pose_aa)
    return local_rots


def convert_clip(clip: dict, motion_rep):
    if "root_trans_offset" not in clip:
        raise ValueError("missing root_trans_offset")

    translation = torch.from_numpy(np.asarray(clip["root_trans_offset"])).float()
    if translation.ndim != 2 or translation.shape[-1] != 3:
        raise ValueError(f"root_trans_offset must have shape [T, 3], got {tuple(translation.shape)}")

    local_rots = clip_to_local_rotations(clip, motion_rep.skeleton.nbjoints)
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


def load_vr_h3_csv(path: Path, root_euler_order: str):
    data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64))
    if data.dtype.names is None:
        raise ValueError("CSV does not have a header")

    required_columns = [
        "root_translateX",
        "root_translateY",
        "root_translateZ",
        "root_rotateX",
        "root_rotateY",
        "root_rotateZ",
        *[joint_name + "_dof" for joint_name in VR_H3_DOF_JOINT_NAMES],
    ]
    missing = [name for name in required_columns if name not in data.dtype.names]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    root_trans = np.stack(
        [data["root_translateX"], data["root_translateY"], data["root_translateZ"]],
        axis=-1,
    ).astype(np.float32) / 100.0
    root_euler_deg = np.stack(
        [data["root_rotateX"], data["root_rotateY"], data["root_rotateZ"]],
        axis=-1,
    )
    root_rot_xyzw = Rotation.from_euler(
        root_euler_order, root_euler_deg, degrees=True
    ).as_quat().astype(np.float32)
    dof_deg = np.stack(
        [data[joint_name + "_dof"] for joint_name in VR_H3_DOF_JOINT_NAMES],
        axis=-1,
    )

    return {
        "root_trans_offset": root_trans,
        "root_rot": root_rot_xyzw,
        "dof": np.deg2rad(dof_deg).astype(np.float32),
        "fps": CSV_DEFAULT_FPS,
    }


def downsample_clip(clip: dict, source_fps: float, target_fps: float):
    if target_fps <= 0 or source_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    stride = max(1, int(round(source_fps / target_fps)))
    actual_fps = source_fps / stride
    if abs(actual_fps - target_fps) > 1e-3:
        raise ValueError(
            f"source_fps={source_fps} cannot be downsampled to target_fps={target_fps} "
            f"with an integer stride"
        )

    return {
        "root_trans_offset": clip["root_trans_offset"][::stride],
        "root_rot": clip["root_rot"][::stride],
        "dof": clip["dof"][::stride],
        "fps": actual_fps,
    }


def root_quat_to_wxyz(root_rot, quat_order: str):
    root_quat = torch.from_numpy(np.asarray(root_rot)).float()
    if quat_order == "xyzw":
        root_quat = root_quat[:, [3, 0, 1, 2]]
    return root_quat


def clip_to_mujoco_qpos(clip: dict, model: mujoco.MjModel, root_quat_order: str):
    root_trans = torch.from_numpy(np.asarray(clip["root_trans_offset"])).float()
    dof = torch.from_numpy(np.asarray(clip["dof"])).float()
    if root_trans.ndim != 2 or root_trans.shape[-1] != 3:
        raise ValueError(f"root_trans_offset must have shape [T, 3], got {tuple(root_trans.shape)}")
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
        qpos[:, model.jnt_qposadr[joint_id]] = dof[:, dof_idx]

    return qpos


def convert_csv_clip(path: Path, motion_rep, model, converter, args):
    clip = load_vr_h3_csv(path, args.csv_root_euler_order)
    clip = downsample_clip(clip, args.source_fps, args.target_fps)
    qpos = clip_to_mujoco_qpos(clip, model, args.root_quat_order)
    if qpos.shape[0] < args.min_frames:
        raise ValueError(f"too few frames after downsampling: {qpos.shape[0]}")

    global_joint_positions, global_joint_rotations = converter.convert_mujoco_qpos_to_motion_transforms(qpos[None])
    full_motion = motion_rep.dual_rep(
        {
            "posed_joints": global_joint_positions[0],
            "global_joint_rots": global_joint_rotations[0],
        },
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


def parse_patterns(patterns: str | None):
    if patterns is None or not patterns.strip():
        return []
    return [pattern.strip().lower() for pattern in patterns.split(",") if pattern.strip()]


def normalize_action_name(stem: str):
    action = stem
    for pattern in ACTION_SUFFIX_PATTERNS:
        action = pattern.sub("", action)
    return action


def load_allow_actions(path: str | None):
    if path is None:
        return None

    actions = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            action = line.split("#", 1)[0].strip()
            if action:
                actions.add(action)
    if not actions:
        raise ValueError(f"allow actions file is empty: {path}")
    return actions


def filter_csv_paths(paths, include_patterns, exclude_patterns, allow_actions):
    filtered = []
    for path in paths:
        name = path.stem.lower()
        if include_patterns and not any(pattern in name for pattern in include_patterns):
            continue
        if exclude_patterns and any(pattern in name for pattern in exclude_patterns):
            continue
        if allow_actions is not None and normalize_action_name(path.stem) not in allow_actions:
            continue
        filtered.append(path)
    return filtered


def compute_stats_streaming(csv_paths, motion_rep, model, converter, args):
    total_frames = 0
    sum_full = torch.zeros(EXPECTED_FULL_DIM, dtype=torch.float64)
    sumsq_full = torch.zeros(EXPECTED_FULL_DIM, dtype=torch.float64)
    valid_paths = []
    failures = []

    for idx, path in enumerate(csv_paths, start=1):
        try:
            full_motion, _ = convert_csv_clip(path, motion_rep, model, converter, args)
            full64 = full_motion.double()
            total_frames += full64.shape[0]
            sum_full += full64.sum(dim=0)
            sumsq_full += (full64 * full64).sum(dim=0)
            valid_paths.append(path)
        except Exception as exc:
            failures.append((str(path), str(exc)))

        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(f"Stats pass: {idx}/{len(csv_paths)} files, valid={len(valid_paths)}, failed={len(failures)}")

    if total_frames == 0:
        raise RuntimeError(f"No frames converted successfully. First failures: {failures[:5]}")

    mean = sum_full / total_frames
    variance = (sumsq_full / total_frames) - (mean * mean)
    std = torch.sqrt(torch.clamp(variance, min=1e-12)).clamp_min(1e-6)
    return mean.float(), std.float(), valid_paths, failures


def build_normalized_dataset_streaming(valid_paths, mean, std, motion_rep, model, converter, args):
    global_indices = motion_rep.dual_rep.indices["global_rep"]
    global_mean = mean[global_indices].float()
    global_std = std[global_indices].float()

    normalized_global_motions = []
    keyids = []
    failures = []

    for idx, path in enumerate(valid_paths, start=1):
        try:
            _, global_motion = convert_csv_clip(path, motion_rep, model, converter, args)
            normalized_global_motions.append(((global_motion - global_mean) / global_std).cpu())
            keyids.append(path.stem)
        except Exception as exc:
            failures.append((str(path), str(exc)))

        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(f"Save pass: {idx}/{len(valid_paths)} files, saved={len(normalized_global_motions)}, failed={len(failures)}")

    if not normalized_global_motions:
        raise RuntimeError(f"No clips saved successfully. First failures: {failures[:5]}")

    return normalized_global_motions, keyids, failures


def main():
    parser = argparse.ArgumentParser(description="Prepare VR H3 CSV data for MotionBricks training")
    parser.add_argument("--input_dir", type=str, default="datas/230424")
    parser.add_argument("--output_dir", type=str, default="datasets/vr_h3_motion_features")
    parser.add_argument("--config", type=str, default="out/motionbricks_vr_h3_vqvae/version_1/hparams.yaml")
    parser.add_argument("--scene_xml", type=str,
                        default="assets/skeletons/vr_h3/mjcf/origin/vr_h3_1_with_rh56e2_hand_scene.xml")
    parser.add_argument("--skeleton_xml", type=str,
                        default="assets/skeletons/vr_h3/mjcf/origin/vr_h3_1_with_rh56e2_hand.xml")
    parser.add_argument("--source_fps", type=float, default=CSV_DEFAULT_FPS)
    parser.add_argument("--target_fps", type=float, default=None,
                        help="Defaults to the FPS in the MotionBricks config.")
    parser.add_argument("--csv_root_euler_order", type=str, default="xyz")
    parser.add_argument("--root_quat_order", choices=["wxyz", "xyzw"], default="xyzw")
    parser.add_argument("--include_patterns", type=str, default=None,
                        help="Comma-separated filename substrings to include, e.g. idle,walk,jog,run,turn.")
    parser.add_argument("--exclude_patterns", type=str, default=None,
                        help="Comma-separated filename substrings to exclude, e.g. jump,dance,combat.")
    parser.add_argument("--allow_actions_file", type=str, default=None,
                        help="Optional newline-separated normalized action names to include exactly.")
    parser.add_argument("--min_frames", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress_every", type=int, default=1000)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    motion_rep = load_vr_h3_motion_rep(args.config)
    if args.target_fps is None:
        args.target_fps = float(motion_rep.fps)

    model = mujoco.MjModel.from_xml_path(args.scene_xml)
    converter = get_mujoco_converter(motion_rep, args.skeleton_xml)

    include_patterns = parse_patterns(args.include_patterns)
    exclude_patterns = parse_patterns(args.exclude_patterns)
    allow_actions = load_allow_actions(args.allow_actions_file)
    csv_paths = sorted(input_dir.rglob("*.csv"))
    total_csv_paths = len(csv_paths)
    csv_paths = filter_csv_paths(csv_paths, include_patterns, exclude_patterns, allow_actions)
    if args.limit is not None:
        csv_paths = csv_paths[:args.limit]
    if not csv_paths:
        raise FileNotFoundError(f"No .csv files found under {input_dir}")

    mean, std, valid_paths, stats_failures = compute_stats_streaming(
        csv_paths, motion_rep, model, converter, args
    )
    normalized_global_motions, keyids, save_failures = build_normalized_dataset_streaming(
        valid_paths, mean, std, motion_rep, model, converter, args
    )
    failures = stats_failures + save_failures

    dataset_path = output_dir / "motions.pt"
    torch.save(
        {
            "motions": normalized_global_motions,
            "keyids": keyids,
            "global_dim": EXPECTED_GLOBAL_DIM,
            "full_dim": EXPECTED_FULL_DIM,
            "source": "csv",
            "source_fps": args.source_fps,
            "target_fps": args.target_fps,
            "include_patterns": include_patterns,
            "exclude_patterns": exclude_patterns,
            "allow_actions_file": args.allow_actions_file,
            "allow_actions": sorted(allow_actions) if allow_actions is not None else None,
        },
        dataset_path,
    )

    stats_dir = output_dir / "stats" / "motion"
    stats_dir.mkdir(parents=True, exist_ok=True)
    np.save(stats_dir / "mean.npy", mean.numpy())
    np.save(stats_dir / "std.npy", std.numpy())

    print(f"Converted clips: {len(normalized_global_motions)}")
    print(f"Failed files: {len(failures)}")
    print(f"CSV files found: {total_csv_paths}")
    print(f"CSV files selected: {len(csv_paths)}")
    if include_patterns:
        print(f"Include patterns: {include_patterns}")
    if exclude_patterns:
        print(f"Exclude patterns: {exclude_patterns}")
    if allow_actions is not None:
        print(f"Allow actions file: {args.allow_actions_file}")
        print(f"Allow actions: {sorted(allow_actions)}")
    print(f"Source FPS: {args.source_fps}")
    print(f"Target FPS: {args.target_fps}")
    print(f"Dataset: {dataset_path}")
    print(f"Stats: {stats_dir}")
    if failures:
        print("First failures:")
        for path, reason in failures[:10]:
            print(f"  {path}: {reason}")


if __name__ == "__main__":
    main()
