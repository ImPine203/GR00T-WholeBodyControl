"""Build a minimal VR H3 clip cache for WASD demo modes."""

import argparse
import os

import torch as t
from omegaconf import OmegaConf, open_dict

from motionbricks.helper.mujoco_helper import get_mujoco_converter
from motionbricks.helper.pl_util import load_motion_rep
from motionbricks.motion_backbone.demo.clips import clip_holder_VR_H3


def load_motion_rep_from_config(config_path: str):
    conf = OmegaConf.load(config_path)
    version_dir = os.path.dirname(config_path)
    with open_dict(conf):
        conf.skeleton.folder = os.path.join(version_dir, "skeleton")
        conf.motion_rep.stats.folder = os.path.join(version_dir, "stats", "motion")
    return load_motion_rep(conf)


def find_motion_index(keyids, requested):
    if isinstance(requested, int):
        return requested
    if requested in keyids:
        return keyids.index(requested)

    matches = [idx for idx, key in enumerate(keyids) if requested.lower() in key.lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"No clip matching '{requested}' found in dataset keyids")
    matched_names = ", ".join(keyids[idx] for idx in matches[:10])
    raise ValueError(f"Clip pattern '{requested}' is ambiguous. First matches: {matched_names}")


def main():
    parser = argparse.ArgumentParser(description="Build VR H3 WASD clip cache")
    parser.add_argument("--dataset_pt", type=str, required=True)
    parser.add_argument("--output", type=str, default="out/VR_H3-clip.ckpt")
    parser.add_argument("--config", type=str, default="out/motionbricks_vr_h3_vqvae/version_1/hparams.yaml")
    parser.add_argument("--skeleton_xml", type=str,
                        default="assets/skeletons/vr_h3/mjcf/origin/vr_h3_1_with_rh56e2_hand.xml")
    parser.add_argument("--idle_clip", type=str, default="hurry_idle_right_R_001__A349")
    parser.add_argument("--slow_walk_clip", type=str, default="walk_180_R_001__A349")
    parser.add_argument("--walk_clip", type=str, default="walk_180_R_003__A349")
    args = parser.parse_args()

    data = t.load(args.dataset_pt, map_location="cpu", weights_only=False)
    motions = data["motions"]
    keyids = data.get("keyids", [])
    if len(keyids) != len(motions):
        raise ValueError("Dataset must contain keyids with the same length as motions")
    clip_source_names = {
        "idle": args.idle_clip,
        "slow_walk": args.slow_walk_clip,
        "walk": args.walk_clip,
    }

    motion_rep = load_motion_rep_from_config(args.config)
    converter = get_mujoco_converter(motion_rep, args.skeleton_xml)
    num_joints = motion_rep.skeleton.nbjoints
    max_num_frames = 20

    clip_items = {}
    for clip_idx, (clip_name, clip_info) in enumerate(clip_holder_VR_H3.CLIPS.items()):
        source_idx = find_motion_index(keyids, clip_source_names[clip_name])
        source_motion = motions[source_idx]
        start_frame = clip_info["start_frame"]
        end_frame = min(clip_info["end_frame"], source_motion.shape[0])
        if end_frame <= start_frame:
            raise ValueError(f"Invalid frame range for {clip_name}: {start_frame}:{end_frame}")

        clip_data = source_motion[None, start_frame:end_frame]
        motion_feature = motion_rep.change_first_heading(
            clip_data, 0.0, is_normalized=True, to_normalize=False
        )[0]

        mujoco_qpos = converter.convert_motion_features_to_mujoco_qpos(
            motion_feature[None], motion_rep, False
        )[0]
        root_rot = mujoco_qpos[:, 3:7].clone()
        mujoco_qpos[:, 3:7] = root_rot[:, [3, 0, 1, 2]]

        global_joint_positions, global_joint_rotations = \
            converter.convert_mujoco_qpos_to_motion_transforms(mujoco_qpos[None])
        global_root_positions = global_joint_positions[0, :, 0] * t.tensor([[1.0, 0.0, 1.0]])
        global_joint_positions = global_joint_positions[0] - global_root_positions[:, None, :]
        global_joint_rotations = global_joint_rotations[0]

        root_direction = t.matmul(
            global_joint_rotations[:, 0, :, :],
            t.tensor([0.0, 0.0, 1.0]).view([1, -1, 1]),
        ).view([-1, 3])
        root_direction = root_direction * t.tensor([1.0, 0.0, 1.0]).view([1, -1])
        root_direction = root_direction / t.norm(root_direction, dim=1, keepdim=True).clamp_min(1e-5)
        global_headings = t.atan2(root_direction[:, 0], root_direction[:, 2])

        clip_items[clip_name] = {
            "global_root_positions": global_root_positions,
            "global_joint_positions": global_joint_positions,
            "global_joint_rotations": global_joint_rotations,
            "global_headings": global_headings,
            "motion_feature": motion_feature,
            "mujoco_qpos": mujoco_qpos,
        }
        max_num_frames = max(max_num_frames, motion_feature.shape[0])
        print(f"{clip_name}: {keyids[source_idx]} frames={motion_feature.shape[0]}")

    state = {}
    buffer_specs = {
        "global_root_positions": [3],
        "global_joint_positions": [num_joints, 3],
        "global_joint_rotations": [num_joints, 3, 3],
        "global_headings": [],
        "motion_feature": [motion_rep.motion_rep_dim],
        "mujoco_qpos": [next(iter(clip_items.values()))["mujoco_qpos"].shape[-1]],
    }
    num_frames_per_clip = t.zeros([len(clip_items)], dtype=t.int32)

    for buffer_name, feat_shape in buffer_specs.items():
        buffer = t.zeros([len(clip_items), max_num_frames, *feat_shape])
        for clip_idx, (clip_name, item) in enumerate(clip_items.items()):
            value = item[buffer_name]
            num_frames_per_clip[clip_idx] = value.shape[0]
            buffer[clip_idx, :value.shape[0]] = value
        state[buffer_name] = buffer

    state["num_frames_per_clip"] = num_frames_per_clip
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    t.save(state, args.output)
    print(f"Saved VR H3 clip cache: {args.output}")


if __name__ == "__main__":
    main()
