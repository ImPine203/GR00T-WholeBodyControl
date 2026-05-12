"""Check VR H3 CSV -> motion features -> MuJoCo qpos reconstruction."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from motionbricks.helper.mujoco_helper import get_mujoco_converter
from prepare_vr_h3_data import (
    clip_to_mujoco_qpos,
    convert_csv_clip,
    downsample_clip,
    load_vr_h3_csv,
    load_vr_h3_motion_rep,
)


def quat_abs_angle_error(q1: torch.Tensor, q2: torch.Tensor):
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    q2 = q2 / q2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    dots = (q1 * q2).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dots)


def hinge_qpos_addresses(model: mujoco.MjModel):
    items = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        items.append((name, int(model.jnt_qposadr[joint_id])))
    return items


def joint_limit_report(model: mujoco.MjModel, qpos: torch.Tensor):
    violations = []
    qpos_np = qpos.detach().cpu().numpy()
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        if not model.jnt_limited[joint_id]:
            continue
        qpos_id = int(model.jnt_qposadr[joint_id])
        values = qpos_np[:, qpos_id]
        low, high = model.jnt_range[joint_id]
        over = np.maximum(np.maximum(low - values, values - high), 0.0)
        max_over = float(over.max())
        if max_over > 1e-6:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            violations.append((max_over, name, float(values.min()), float(values.max()), float(low), float(high)))
    violations.sort(reverse=True)
    return violations


def body_positions(model: mujoco.MjModel, qpos: torch.Tensor, body_names: list[str]):
    data = mujoco.MjData(model)
    body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in body_names
    ]
    missing = [name for name, body_id in zip(body_names, body_ids) if body_id < 0]
    if missing:
        raise ValueError(f"Missing bodies: {missing}")

    qpos_np = qpos.detach().cpu().numpy()
    out = np.zeros((qpos_np.shape[0], len(body_ids), 3), dtype=np.float64)
    for frame_idx, frame in enumerate(qpos_np):
        data.qpos[:] = frame
        mujoco.mj_forward(model, data)
        for body_idx, body_id in enumerate(body_ids):
            out[frame_idx, body_idx] = data.xpos[body_id]
    return torch.from_numpy(out).float()


def summarize_errors(name: str, value: torch.Tensor):
    value = value.detach().cpu().float()
    print(
        f"{name}: mean={value.mean().item():.8f}, "
        f"p95={torch.quantile(value, 0.95).item():.8f}, "
        f"max={value.max().item():.8f}"
    )


def play_qpos(model: mujoco.MjModel, qpos: torch.Tensor, fps: float, speed: float, loop: bool):
    data = mujoco.MjData(model)
    qpos_np = qpos.detach().cpu().numpy()
    frame_dt = 1.0 / fps / speed

    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame = 0
        while viewer.is_running():
            start = time.time()
            data.qpos[:] = qpos_np[frame]
            mujoco.mj_forward(model, data)
            viewer.cam.lookat[:] = data.qpos[:3]
            viewer.sync()

            frame += 1
            if frame >= qpos_np.shape[0]:
                if not loop:
                    break
                frame = 0

            sleep_time = frame_dt - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)

from scipy.spatial.transform import Rotation as R
import numpy as np
import torch


def wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    return q_wxyz[..., [1, 2, 3, 0]]


def xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    return q_xyzw[..., [3, 0, 1, 2]]


def yaw_z_from_wxyz(q_wxyz: np.ndarray) -> float:
    q_xyzw = wxyz_to_xyzw(q_wxyz)
    return R.from_quat(q_xyzw).as_euler("xyz", degrees=False)[2]


def align_reconstructed_qpos_to_source(
    qpos_src: torch.Tensor,
    qpos_rec: torch.Tensor,
) -> torch.Tensor:
    """
    Align reconstructed qpos from canonical/root-centered frame
    back to the source motion's initial world root pose.

    Assumes MuJoCo qpos quaternion is wxyz and world is Z-up.
    """
    src = qpos_src.detach().cpu().numpy()
    rec = qpos_rec.detach().cpu().numpy()

    src_pos0 = src[0, :3].copy()
    rec_pos0 = rec[0, :3].copy()

    src_yaw0 = yaw_z_from_wxyz(src[0, 3:7])
    rec_yaw0 = yaw_z_from_wxyz(rec[0, 3:7])

    yaw_delta = src_yaw0 - rec_yaw0
    rot_align = R.from_euler("z", yaw_delta, degrees=False)

    out = rec.copy()

    # Align root positions.
    # Rotate reconstructed trajectory around its first root position,
    # then translate it to source first root position.
    out[:, :3] = rot_align.apply(rec[:, :3] - rec_pos0) + src_pos0

    # Align root orientations.
    rec_quat_xyzw = wxyz_to_xyzw(rec[:, 3:7])
    rec_rots = R.from_quat(rec_quat_xyzw)
    aligned_rots = rot_align * rec_rots
    out[:, 3:7] = xyzw_to_wxyz(aligned_rots.as_quat())

    return torch.from_numpy(out).to(qpos_rec.device, dtype=qpos_rec.dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--config", type=str, default="out/motionbricks_vr_h3_vqvae/version_1/hparams.yaml")
    parser.add_argument("--scene_xml", type=str,
                        default="assets/skeletons/vr_h3/mjcf/origin/vr_h3_1_with_rh56e2_hand_scene.xml")
    parser.add_argument("--skeleton_xml", type=str,
                        default="assets/skeletons/vr_h3/mjcf/origin/vr_h3_1_with_rh56e2_hand.xml")
    parser.add_argument("--source_fps", type=float, default=120.0)
    parser.add_argument("--target_fps", type=float, default=None)
    parser.add_argument("--csv_root_euler_order", type=str, default="xyz")
    parser.add_argument("--root_quat_order", choices=["wxyz", "xyzw"], default="xyzw")
    parser.add_argument("--min_frames", type=int, default=8)
    parser.add_argument("--play", choices=["none", "source", "reconstructed"], default="none",
                        help="Open MuJoCo viewer for the source qpos or inverse-feature reconstructed qpos.")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    motion_rep = load_vr_h3_motion_rep(args.config)
    if args.target_fps is None:
        args.target_fps = float(motion_rep.fps)

    model = mujoco.MjModel.from_xml_path(args.scene_xml)
    converter = get_mujoco_converter(motion_rep, args.skeleton_xml)

    clip = load_vr_h3_csv(args.csv, args.csv_root_euler_order)
    clip = downsample_clip(clip, args.source_fps, args.target_fps)
    qpos_src = clip_to_mujoco_qpos(clip, model, args.root_quat_order)

    _, global_motion = convert_csv_clip(args.csv, motion_rep, model, converter, args)
    qpos_rec = converter.convert_motion_features_to_mujoco_qpos(
        global_motion[None], motion_rep, False
    )[0]
    root_quat = qpos_rec[:, 3:7].clone()
    qpos_rec[:, 3:7] = root_quat[:, [3, 0, 1, 2]]

    # NEW: align canonical reconstructed motion to source initial root pose
    qpos_rec_raw = qpos_rec.clone()
    qpos_rec = align_reconstructed_qpos_to_source(qpos_src, qpos_rec_raw)

    frame_count = min(qpos_src.shape[0], qpos_rec.shape[0])
    qpos_src = qpos_src[:frame_count]
    qpos_rec = qpos_rec[:frame_count]

    root_pos_err = (qpos_src[:, :3] - qpos_rec[:, :3]).norm(dim=-1)
    root_rot_err = quat_abs_angle_error(qpos_src[:, 3:7], qpos_rec[:, 3:7])

    hinge_items = hinge_qpos_addresses(model)
    hinge_addrs = torch.tensor([addr for _, addr in hinge_items], dtype=torch.long)
    joint_err = (qpos_src[:, hinge_addrs] - qpos_rec[:, hinge_addrs]).abs()
    worst_joint_err, worst_joint_idx = joint_err.max(dim=0)
    worst_order = torch.argsort(worst_joint_err, descending=True)

    feet_src = body_positions(model, qpos_src, ["left_ankle_roll_link", "right_ankle_roll_link"])
    feet_rec = body_positions(model, qpos_rec, ["left_ankle_roll_link", "right_ankle_roll_link"])
    foot_pos_err = (feet_src - feet_rec).norm(dim=-1)

    src_delta = qpos_src[-1, :3] - qpos_src[0, :3]
    rec_delta = qpos_rec[-1, :3] - qpos_rec[0, :3]

    src_dir = src_delta / src_delta.norm().clamp_min(1e-8)
    rec_dir = rec_delta / rec_delta.norm().clamp_min(1e-8)

    print("src first root:", qpos_src[0, :3])
    print("src last  root:", qpos_src[-1, :3])
    print("rec first root:", qpos_rec[0, :3])
    print("rec last  root:", qpos_rec[-1, :3])
    print("src_delta:", src_delta)
    print("rec_delta:", rec_delta)
    print("src_dir:", src_dir)
    print("rec_dir:", rec_dir)

    print("== VR H3 feature roundtrip ==")
    print(f"csv: {args.csv}")
    print(f"frames: {frame_count}")
    print(f"global_motion_shape: {tuple(global_motion.shape)}")
    print(f"qpos_shape: {tuple(qpos_src.shape)}")
    print(f"finite_src: {torch.isfinite(qpos_src).all().item()}")
    print(f"finite_rec: {torch.isfinite(qpos_rec).all().item()}")
    summarize_errors("root_pos_err_m", root_pos_err)
    summarize_errors("root_rot_err_rad", root_rot_err)
    summarize_errors("joint_abs_err_rad", joint_err.reshape(-1))
    summarize_errors("foot_pos_err_m", foot_pos_err.reshape(-1))

    print("worst_joint_abs_err_rad:")
    for idx in worst_order[:10].tolist():
        name = hinge_items[idx][0]
        print(f"  {name}: max={worst_joint_err[idx].item():.8f}")

    violations = joint_limit_report(model, qpos_rec)
    if violations:
        print("reconstructed_joint_limit_violations:")
        for over, name, min_value, max_value, low, high in violations[:10]:
            print(
                f"  {name}: over={over:.8f}, value_range=[{min_value:.8f}, {max_value:.8f}], "
                f"limit=[{low:.8f}, {high:.8f}]"
            )
    else:
        print("reconstructed_joint_limit_violations: none")

    if args.play != "none":
        qpos_to_play = qpos_src if args.play == "source" else qpos_rec
        print(f"Playing {args.play} qpos at {args.target_fps:g} FPS, speed={args.speed:g}")
        play_qpos(model, qpos_to_play, args.target_fps, args.speed, args.loop)


if __name__ == "__main__":
    main()
