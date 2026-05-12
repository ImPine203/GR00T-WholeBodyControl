"""Audit VR H3 CSV files before using them as MotionBricks training input."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np
import torch

from motionbricks.helper.mujoco_helper import get_mujoco_converter
from check_vr_h3_feature_roundtrip import (
    body_positions,
    hinge_qpos_addresses,
    joint_limit_report,
    quat_abs_angle_error,
)
from prepare_vr_h3_data import (
    clip_to_mujoco_qpos,
    convert_csv_clip,
    downsample_clip,
    filter_csv_paths,
    load_allow_actions,
    load_vr_h3_csv,
    load_vr_h3_motion_rep,
    parse_patterns,
)


def summarize(values: list[float]):
    if not values:
        return {"mean": np.nan, "p95": np.nan, "max": np.nan}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def root_stats(qpos: torch.Tensor, fps: float):
    root = qpos[:, :3]
    duration = max((qpos.shape[0] - 1) / fps, 1e-6)
    delta = root[-1] - root[0]
    return {
        "frames": int(qpos.shape[0]),
        "duration_s": float(duration),
        "root_delta_x_m": float(delta[0]),
        "root_delta_y_m": float(delta[1]),
        "root_delta_z_m": float(delta[2]),
        "root_speed_xy_mps": float(torch.linalg.norm(delta[:2]).item() / duration),
        "root_height_mean_m": float(root[:, 2].mean()),
        "root_height_min_m": float(root[:, 2].min()),
        "root_height_max_m": float(root[:, 2].max()),
    }


def max_joint_limit_violation(violations):
    return float(violations[0][0]) if violations else 0.0


def audit_clip(path: Path, motion_rep, model, converter, args):
    clip = load_vr_h3_csv(path, args.csv_root_euler_order)
    clip = downsample_clip(clip, args.source_fps, args.target_fps)
    qpos_src = clip_to_mujoco_qpos(clip, model, args.root_quat_order)

    _, global_motion = convert_csv_clip(path, motion_rep, model, converter, args)
    qpos_rec = converter.convert_motion_features_to_mujoco_qpos(
        global_motion[None], motion_rep, False
    )[0]
    root_quat = qpos_rec[:, 3:7].clone()
    qpos_rec[:, 3:7] = root_quat[:, [3, 0, 1, 2]]

    frame_count = min(qpos_src.shape[0], qpos_rec.shape[0])
    qpos_src = qpos_src[:frame_count]
    qpos_rec = qpos_rec[:frame_count]

    hinge_items = hinge_qpos_addresses(model)
    hinge_addrs = torch.tensor([addr for _, addr in hinge_items], dtype=torch.long)
    joint_err = (qpos_src[:, hinge_addrs] - qpos_rec[:, hinge_addrs]).abs()
    root_pos_err = (qpos_src[:, :3] - qpos_rec[:, :3]).norm(dim=-1)
    root_rot_err = quat_abs_angle_error(qpos_src[:, 3:7], qpos_rec[:, 3:7])
    feet_src = body_positions(model, qpos_src, ["left_ankle_roll_link", "right_ankle_roll_link"])
    feet_rec = body_positions(model, qpos_rec, ["left_ankle_roll_link", "right_ankle_roll_link"])
    foot_pos_err = (feet_src - feet_rec).norm(dim=-1)

    src_violations = joint_limit_report(model, qpos_src)
    rec_violations = joint_limit_report(model, qpos_rec)
    stats = root_stats(qpos_src, args.target_fps)
    stats.update(
        {
            "path": str(path),
            "keyid": path.stem,
            "global_dim": int(global_motion.shape[-1]),
            "finite_src": bool(torch.isfinite(qpos_src).all()),
            "finite_rec": bool(torch.isfinite(qpos_rec).all()),
            "root_pos_err_mean_m": float(root_pos_err.mean()),
            "root_pos_err_p95_m": float(torch.quantile(root_pos_err, 0.95)),
            "root_rot_err_mean_rad": float(root_rot_err.mean()),
            "root_rot_err_p95_rad": float(torch.quantile(root_rot_err, 0.95)),
            "joint_abs_err_max_rad": float(joint_err.max()),
            "foot_pos_err_mean_m": float(foot_pos_err.mean()),
            "src_joint_limit_max_over_rad": max_joint_limit_violation(src_violations),
            "rec_joint_limit_max_over_rad": max_joint_limit_violation(rec_violations),
            "src_joint_limit_count": len(src_violations),
            "rec_joint_limit_count": len(rec_violations),
        }
    )
    return stats


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict], failures: list[tuple[str, str]], args, total_csv: int, selected_csv: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "root_pos_err_mean_m",
        "root_rot_err_mean_rad",
        "joint_abs_err_max_rad",
        "foot_pos_err_mean_m",
        "src_joint_limit_max_over_rad",
        "root_speed_xy_mps",
        "root_delta_y_m",
    ]
    with path.open("w") as f:
        f.write("# VR H3 Training Input Audit\n\n")
        f.write(f"- Input dir: `{args.input_dir}`\n")
        f.write(f"- CSV files found: `{total_csv}`\n")
        f.write(f"- CSV files selected: `{selected_csv}`\n")
        f.write(f"- CSV files audited: `{len(rows)}`\n")
        f.write(f"- Failures: `{len(failures)}`\n")
        f.write(f"- Allow actions file: `{args.allow_actions_file}`\n")
        f.write(f"- Include patterns: `{parse_patterns(args.include_patterns)}`\n")
        f.write(f"- Exclude patterns: `{parse_patterns(args.exclude_patterns)}`\n\n")

        f.write("## Metrics\n\n")
        for field in fields:
            stats = summarize([float(row[field]) for row in rows])
            f.write(
                f"- `{field}`: mean={stats['mean']:.6g}, "
                f"p95={stats['p95']:.6g}, max={stats['max']:.6g}\n"
            )

        bad_root = [row for row in rows if float(row["root_rot_err_mean_rad"]) > args.max_root_rot_err]
        bad_joint = [row for row in rows if float(row["joint_abs_err_max_rad"]) > args.max_joint_err]
        bad_limit = [row for row in rows if float(row["src_joint_limit_max_over_rad"]) > args.max_joint_limit_over]
        f.write("\n## Gates\n\n")
        f.write(f"- Root rotation err > {args.max_root_rot_err}: `{len(bad_root)}` clips\n")
        f.write(f"- Joint roundtrip err > {args.max_joint_err}: `{len(bad_joint)}` clips\n")
        f.write(f"- Source joint-limit over > {args.max_joint_limit_over}: `{len(bad_limit)}` clips\n")

        if failures:
            f.write("\n## First Failures\n\n")
            for failed_path, reason in failures[:10]:
                f.write(f"- `{failed_path}`: {reason}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=str, default="datas/vr_h3_data/vr_h3_1")
    parser.add_argument("--output_dir", type=str, default="out/vr_h3_training_input_audit")
    parser.add_argument("--config", type=str, default="out/motionbricks_vr_h3_vqvae/version_1/hparams.yaml")
    parser.add_argument("--scene_xml", type=str,
                        default="assets/skeletons/vr_h3/mjcf/origin/vr_h3_1_with_rh56e2_hand_scene.xml")
    parser.add_argument("--skeleton_xml", type=str,
                        default="assets/skeletons/vr_h3/mjcf/origin/vr_h3_1_with_rh56e2_hand.xml")
    parser.add_argument("--source_fps", type=float, default=120.0)
    parser.add_argument("--target_fps", type=float, default=None)
    parser.add_argument("--csv_root_euler_order", type=str, default="xyz")
    parser.add_argument("--root_quat_order", choices=["wxyz", "xyzw"], default="xyzw")
    parser.add_argument("--include_patterns", type=str, default=None)
    parser.add_argument("--exclude_patterns", type=str, default=None)
    parser.add_argument("--allow_actions_file", type=str, default=None)
    parser.add_argument("--min_frames", type=int, default=8)
    parser.add_argument("--limit", type=int, default=100,
                        help="Maximum selected CSV files to audit. Use 0 to audit all selected files.")
    parser.add_argument("--max_root_rot_err", type=float, default=0.25)
    parser.add_argument("--max_joint_err", type=float, default=1e-4)
    parser.add_argument("--max_joint_limit_over", type=float, default=1e-3)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    motion_rep = load_vr_h3_motion_rep(args.config)
    if args.target_fps is None:
        args.target_fps = float(motion_rep.fps)

    include_patterns = parse_patterns(args.include_patterns)
    exclude_patterns = parse_patterns(args.exclude_patterns)
    allow_actions = load_allow_actions(args.allow_actions_file)
    csv_paths = sorted(input_dir.rglob("*.csv"))
    total_csv = len(csv_paths)
    selected_paths = filter_csv_paths(csv_paths, include_patterns, exclude_patterns, allow_actions)
    selected_csv = len(selected_paths)
    if args.limit and args.limit > 0:
        selected_paths = selected_paths[:args.limit]

    model = mujoco.MjModel.from_xml_path(args.scene_xml)
    converter = get_mujoco_converter(motion_rep, args.skeleton_xml)

    rows = []
    failures = []
    for idx, path in enumerate(selected_paths, start=1):
        try:
            rows.append(audit_clip(path, motion_rep, model, converter, args))
        except Exception as exc:
            failures.append((str(path), str(exc)))
        if idx % 25 == 0 or idx == len(selected_paths):
            print(f"Audited {idx}/{len(selected_paths)} files, ok={len(rows)}, failed={len(failures)}")

    report_csv = output_dir / "clips.csv"
    summary_md = output_dir / "summary.md"
    write_csv(report_csv, rows)
    write_summary(summary_md, rows, failures, args, total_csv, selected_csv)

    print("== VR H3 training input audit ==")
    print(f"CSV files found: {total_csv}")
    print(f"CSV files selected: {selected_csv}")
    print(f"CSV files audited: {len(rows)}")
    print(f"Failed files: {len(failures)}")
    print(f"Report: {report_csv}")
    print(f"Summary: {summary_md}")


if __name__ == "__main__":
    main()
