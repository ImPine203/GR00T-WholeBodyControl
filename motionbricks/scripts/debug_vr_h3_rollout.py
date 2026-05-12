"""Headless VR H3 rollout debug for WASD-generated motion."""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

os.environ.setdefault("PYNPUT_BACKEND", "dummy")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import mujoco
import numpy as np
import torch as t

from motionbricks.motion_backbone.demo.utils import navigation_demo


CONTROL_KEYS = [
    "w", "a", "s", "d",
    "left", "right", "up", "down",
    "shift", "ctrl", "enter",
    "x", "z", "c", "v", "b", "r", "t", "f", "g", "q", "e",
]


def make_key_pressed(active_key: str | None):
    key_pressed = {key: False for key in CONTROL_KEYS}
    if active_key:
        for key in active_key.split(","):
            key = key.strip()
            if key:
                key_pressed[key] = True
    return key_pressed


def build_args(args):
    speed_scale = [float(i) for i in args.speed_scale.split(",")]
    return SimpleNamespace(
        humanoid_xml=args.humanoid_xml,
        humanoid_scene_xml=args.humanoid_xml,
        skeleton_xml=args.skeleton_xml,
        clips_ckpt=args.clips_ckpt,
        result_dir=args.result_dir,
        data_root=args.data_root,
        explicit_dataset_folder=None,
        reprocess_clips=0,
        controller="wasd",
        lookat_movement_direction=args.lookat_movement_direction,
        pre_filter_qpos=args.pre_filter_qpos,
        source_root_realignment=args.source_root_realignment,
        target_root_realignment=args.target_root_realignment,
        force_canonicalization=args.force_canonicalization,
        skip_ending_target_cond=args.skip_ending_target_cond,
        random_speed_scale=1,
        speed_scale=speed_scale,
        generate_dt=args.generate_dt,
        use_qpos=1,
        planner="vr_h3",
        EXP="vr_h3",
        allowed_mode=None,
        clips="vr_h3",
        return_model_configs=True,
        return_dataloader=True,
        recording_dir=None,
    )


def hinge_joint_limits(model: mujoco.MjModel):
    items = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        if not model.jnt_limited[joint_id]:
            continue
        qpos_adr = model.jnt_qposadr[joint_id]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        low, high = model.jnt_range[joint_id]
        items.append((name, qpos_adr, float(low), float(high)))
    return items


def joint_limit_report(qpos_frames: np.ndarray, limits):
    worst = []
    max_violation = 0.0
    violation_frames = 0
    for name, qpos_adr, low, high in limits:
        values = qpos_frames[:, qpos_adr]
        below = low - values
        above = values - high
        violation = np.maximum(np.maximum(below, above), 0.0)
        joint_max = float(violation.max())
        if joint_max > 0.0:
            violation_frames += int((violation > 0.0).sum())
            worst.append((joint_max, name, float(values.min()), float(values.max()), low, high))
            max_violation = max(max_violation, joint_max)
    worst.sort(reverse=True)
    return max_violation, violation_frames, worst[:10]


def main():
    parser = argparse.ArgumentParser(description="Debug VR H3 generated rollout without opening a viewer")
    parser.add_argument("--humanoid_xml", type=str,
                        default="assets/skeletons/vr_h3/mjcf/origin/vr_h3_1_with_rh56e2_hand_scene.xml")
    parser.add_argument("--skeleton_xml", type=str,
                        default="assets/skeletons/vr_h3/mjcf/origin/vr_h3_1_with_rh56e2_hand.xml")
    parser.add_argument("--clips_ckpt", type=str, default="out/VR_H3-clean-forward-walk-clip.ckpt")
    parser.add_argument("--result_dir", type=str, default="./out")
    parser.add_argument("--data_root", type=str, default="./datasets")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--warmup_steps", type=int, default=30)
    parser.add_argument("--key", type=str, default="w")
    parser.add_argument("--save_npz", type=str, default="out/vr_h3_debug_rollout_w.npz")
    parser.add_argument("--lookat_movement_direction", type=int, default=1)
    parser.add_argument("--pre_filter_qpos", type=int, default=1)
    parser.add_argument("--source_root_realignment", type=int, default=1)
    parser.add_argument("--target_root_realignment", type=int, default=1)
    parser.add_argument("--force_canonicalization", type=int, default=1)
    parser.add_argument("--skip_ending_target_cond", type=int, default=0)
    parser.add_argument("--speed_scale", type=str, default="1.0,1.0")
    parser.add_argument("--target_vel", type=float, default=None,
                        help="Optional target root speed in m/s for non-idle modes.")
    parser.add_argument("--generate_dt", type=float, default=2.0)
    args = parser.parse_args()

    demo_args = build_args(args)
    demo_agent = navigation_demo(demo_args)
    demo_agent.full_agent.reset()

    qpos_frames = []
    modes = []
    key_idle = make_key_pressed(None)
    key_active = make_key_pressed(args.key)

    for step in range(args.steps):
        qpos = demo_agent.full_agent.get_next_frame()
        context_mujoco_qpos = demo_agent.full_agent.get_context_mujoco_qpos()
        if isinstance(qpos, t.Tensor):
            qpos = qpos.detach().cpu().numpy()
        demo_agent.mj_data.qpos[:] = qpos

        key_pressed = key_idle if step < args.warmup_steps else key_active
        control_signals = demo_agent.controller.generate_control_signals(
            None,
            demo_agent.mj_model,
            demo_agent.mj_data,
            visualize=False,
            control_info={"force_idle": False, "allowed_mode": None, "key_pressed": key_pressed},
        )
        control_signals["context_mujoco_qpos"] = context_mujoco_qpos
        if args.target_vel is not None and args.target_vel > 0.0:
            # full_agent multiplies target_vel by 2 internally.
            control_signals["target_vel"] = t.tensor([args.target_vel / 2.0], dtype=t.float32)
        modes.append(int(control_signals["mode"].view(-1)[0].item()))

        with t.no_grad():
            demo_agent.full_agent.generate_new_frames(
                control_signals,
                demo_agent.controller.get_controller_dt() * args.generate_dt,
            )
        mujoco.mj_forward(demo_agent.mj_model, demo_agent.mj_data)
        qpos_frames.append(demo_agent.mj_data.qpos.copy())

    qpos_frames = np.stack(qpos_frames)
    root_delta = qpos_frames[-1, :3] - qpos_frames[0, :3]
    active_delta = qpos_frames[-1, :3] - qpos_frames[args.warmup_steps, :3]
    finite = np.isfinite(qpos_frames).all()
    max_violation, violation_frames, worst = joint_limit_report(qpos_frames, hinge_joint_limits(demo_agent.mj_model))

    print("== VR H3 rollout debug ==")
    print(f"steps: {args.steps}, warmup_steps: {args.warmup_steps}, key: {args.key}")
    print(f"qpos shape: {qpos_frames.shape}, finite: {finite}")
    print(f"root_delta_total_xyz: {root_delta.tolist()}")
    print(f"root_delta_after_warmup_xyz: {active_delta.tolist()}")
    print(f"mode ids seen: {sorted(set(modes))}")
    print(f"max_joint_limit_violation_rad: {max_violation}")
    print(f"joint_limit_violation_frames: {violation_frames}")
    if worst:
        print("worst_joint_limit_violations:")
        for joint_max, name, vmin, vmax, low, high in worst:
            print(f"  {name}: violation={joint_max:.6f}, value_range=[{vmin:.6f}, {vmax:.6f}], limit=[{low:.6f}, {high:.6f}]")

    if args.save_npz:
        np.savez(args.save_npz, qpos=qpos_frames, modes=np.asarray(modes), root_delta=root_delta)
        print(f"saved: {args.save_npz}")


if __name__ == "__main__":
    main()
