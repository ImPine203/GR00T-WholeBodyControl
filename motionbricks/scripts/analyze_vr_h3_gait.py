"""Analyze VR H3 foot motion from rollout qpos and primitive clip cache."""

from __future__ import annotations

import argparse

import mujoco
import numpy as np
import torch


def foot_positions(model, qpos_frames, foot_body_names):
    data = mujoco.MjData(model)
    body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in foot_body_names
    ]
    if any(body_id < 0 for body_id in body_ids):
        missing = [name for name, body_id in zip(foot_body_names, body_ids) if body_id < 0]
        raise ValueError(f"Missing foot bodies: {missing}")

    positions = np.zeros((qpos_frames.shape[0], len(body_ids), 3), dtype=np.float64)
    for frame_idx, qpos in enumerate(qpos_frames):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        for body_idx, body_id in enumerate(body_ids):
            positions[frame_idx, body_idx] = data.xpos[body_id]
    return positions


def summarize(name, qpos_frames, model, fps):
    feet = foot_positions(model, qpos_frames, ["left_ankle_roll_link", "right_ankle_roll_link"])
    left = feet[:, 0]
    right = feet[:, 1]

    left_height = left[:, 2]
    right_height = right[:, 2]
    left_forward = left[:, 0]
    right_forward = right[:, 0]
    left_speed = np.linalg.norm(np.diff(left[:, :2], axis=0), axis=1) * fps
    right_speed = np.linalg.norm(np.diff(right[:, :2], axis=0), axis=1) * fps
    root_forward = qpos_frames[:, 0]
    root_lateral = qpos_frames[:, 1]

    left_lift = left_height - np.percentile(left_height, 10)
    right_lift = right_height - np.percentile(right_height, 10)
    lift_corr = float(np.corrcoef(left_lift, right_lift)[0, 1]) if len(left_lift) > 1 else float("nan")
    step_asym = float(abs(left_forward[-1] - left_forward[0] - (right_forward[-1] - right_forward[0])))

    print(f"== {name} ==")
    print(f"frames: {qpos_frames.shape[0]}")
    print(f"root_delta_xy: {[float(root_forward[-1] - root_forward[0]), float(root_lateral[-1] - root_lateral[0])]}")
    print(f"left_height_range:  {[float(left_height.min()), float(left_height.max())]}")
    print(f"right_height_range: {[float(right_height.min()), float(right_height.max())]}")
    print(f"left_lift_p95:  {float(np.percentile(left_lift, 95))}")
    print(f"right_lift_p95: {float(np.percentile(right_lift, 95))}")
    print(f"left_speed_mean/max:  {[float(left_speed.mean()), float(left_speed.max())]}")
    print(f"right_speed_mean/max: {[float(right_speed.mean()), float(right_speed.max())]}")
    print(f"left_forward_delta:  {float(left_forward[-1] - left_forward[0])}")
    print(f"right_forward_delta: {float(right_forward[-1] - right_forward[0])}")
    print(f"forward_step_asymmetry: {step_asym}")
    print(f"left_right_lift_corr: {lift_corr}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze VR H3 gait foot motion")
    parser.add_argument("--rollout_npz", type=str, default="out/vr_h3_debug_rollout_w.npz")
    parser.add_argument("--cache", type=str, default="out/VR_H3-clean-forward-walk-clip.ckpt")
    parser.add_argument("--scene_xml", type=str,
                        default="assets/skeletons/vr_h3/mjcf/origin/vr_h3_1_with_rh56e2_hand_scene.xml")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--warmup_steps", type=int, default=30)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene_xml)
    rollout = np.load(args.rollout_npz)["qpos"]
    summarize("generated rollout after warmup", rollout[args.warmup_steps:], model, args.fps)

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    names = ["idle", "slow_walk", "walk"]
    for idx, name in enumerate(names):
        frames = int(cache["num_frames_per_clip"][idx])
        qpos = cache["mujoco_qpos"][idx, :frames].numpy()
        summarize(f"primitive {name}", qpos, model, args.fps)


if __name__ == "__main__":
    main()
