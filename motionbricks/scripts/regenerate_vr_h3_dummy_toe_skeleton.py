"""Regenerate VR H3 neutral skeleton files with dummy toe joints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from motionbricks.motionlib.core.skeletons.vr_h3 import H3Skeleton


OLD_BONE_ORDER_NAMES = [
    "pelvis_skel",
    "left_hip_pitch_skel",
    "left_hip_roll_skel",
    "left_hip_yaw_skel",
    "left_knee_pitch_skel",
    "left_ankle_pitch_skel",
    "left_ankle_roll_skel",
    "right_hip_pitch_skel",
    "right_hip_roll_skel",
    "right_hip_yaw_skel",
    "right_knee_pitch_skel",
    "right_ankle_pitch_skel",
    "right_ankle_roll_skel",
    "waist_yaw_skel",
    "waist_roll_skel",
    "left_shoulder_pitch_skel",
    "left_shoulder_roll_skel",
    "left_shoulder_yaw_skel",
    "left_elbow_pitch_skel",
    "left_wrist_yaw_skel",
    "left_wrist_roll_skel",
    "left_wrist_pitch_skel",
    "right_shoulder_pitch_skel",
    "right_shoulder_roll_skel",
    "right_shoulder_yaw_skel",
    "right_elbow_pitch_skel",
    "right_wrist_yaw_skel",
    "right_wrist_roll_skel",
    "right_wrist_pitch_skel",
    "head_yaw_skel",
    "head_pitch_skel",
]


DEFAULT_SKELETON_DIRS = [
    Path("out/motionbricks_vr_h3_vqvae/version_1/skeleton"),
    Path("out/motionbricks_vr_h3_pose/version_1/skeleton"),
    Path("out/motionbricks_vr_h3_root/version_1/skeleton"),
]
EXPECTED_FULL_DIM = 406


def build_new_skeleton(old_joints: torch.Tensor, toe_offset: torch.Tensor):
    old_joints = old_joints.squeeze()
    new_names = [name for name, _ in H3Skeleton.bone_order_names_with_parents]
    if old_joints.shape == (len(OLD_BONE_ORDER_NAMES), 3):
        old_names = OLD_BONE_ORDER_NAMES
    elif old_joints.shape == (len(new_names), 3):
        old_names = new_names
    else:
        raise ValueError(
            f"Expected old joints shape {(len(OLD_BONE_ORDER_NAMES), 3)} or {(len(new_names), 3)}, "
            f"got {tuple(old_joints.shape)}"
        )

    old_by_name = dict(zip(old_names, old_joints))
    new_joints = []
    for name in new_names:
        if name == "left_toe_base":
            new_joints.append(old_by_name["left_ankle_roll_skel"] + toe_offset)
        elif name == "right_toe_base":
            new_joints.append(old_by_name["right_ankle_roll_skel"] + toe_offset)
        else:
            new_joints.append(old_by_name[name])

    name_to_idx = {name: idx for idx, name in enumerate(new_names)}
    new_parents = torch.tensor(
        [-1 if parent is None else name_to_idx[parent] for name, parent in H3Skeleton.bone_order_names_with_parents],
        dtype=torch.long,
    )
    return torch.stack(new_joints).to(old_joints.dtype), new_parents


def regenerate_dir(skeleton_dir: Path, toe_offset: torch.Tensor, dry_run: bool):
    joints_path = skeleton_dir / "joints.p"
    parents_path = skeleton_dir / "parents.p"
    old_joints = torch.load(joints_path, map_location="cpu")
    new_joints, new_parents = build_new_skeleton(old_joints, toe_offset)

    if dry_run:
        print(f"{skeleton_dir}: would write joints={tuple(new_joints.shape)}, parents={tuple(new_parents.shape)}")
        return

    torch.save(new_joints, joints_path)
    torch.save(new_parents, parents_path)
    print(f"{skeleton_dir}: wrote joints={tuple(new_joints.shape)}, parents={tuple(new_parents.shape)}")


def reset_stats_for_skeleton_dir(skeleton_dir: Path, dry_run: bool):
    stats_dir = skeleton_dir.parent / "stats" / "motion"
    if dry_run:
        print(f"{stats_dir}: would write placeholder mean/std dim={EXPECTED_FULL_DIM}")
        return

    stats_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.save(stats_dir / "mean.npy", np.zeros(EXPECTED_FULL_DIM, dtype=np.float32))
    np.save(stats_dir / "std.npy", np.ones(EXPECTED_FULL_DIM, dtype=np.float32))
    print(f"{stats_dir}: wrote placeholder mean/std dim={EXPECTED_FULL_DIM}")


def main():
    parser = argparse.ArgumentParser(description="Add VR H3 dummy toe joints to MotionBricks skeleton files")
    parser.add_argument("--skeleton_dir", action="append", type=Path, default=None)
    parser.add_argument("--toe_offset", nargs=3, type=float, default=[0.0, -0.035, 0.14],
                        metavar=("X", "Y", "Z"),
                        help="Toe offset from ankle roll in motion space. Motion space is Y-up, Z-forward.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--reset_stats", action="store_true",
                        help="Write placeholder 406-dim stats so the new 33-joint motion_rep can instantiate before data prepare.")
    args = parser.parse_args()

    skeleton_dirs = args.skeleton_dir or DEFAULT_SKELETON_DIRS
    toe_offset = torch.tensor(args.toe_offset, dtype=torch.float32)
    for skeleton_dir in skeleton_dirs:
        regenerate_dir(skeleton_dir, toe_offset, args.dry_run)
        if args.reset_stats:
            reset_stats_for_skeleton_dir(skeleton_dir, args.dry_run)


if __name__ == "__main__":
    main()
