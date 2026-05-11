"""Root model training script using synthetic data.

Demonstrates how the root backbone training pipeline works without
requiring the actual motion dataset. Loads the saved model config from
the checkpoint directory and trains on randomly generated motion tensors.

The root model does not require a pretrained VQVAE — it directly
predicts continuous root motion values.

Usage:
    python scripts/train_root.py --max_steps 100
"""

import argparse
import copy
import os
import shutil

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from motionbricks.data.synthetic_dataset import SyntheticMotionDataset, TorchMotionDataset, collate_batch
from motionbricks.helper.pl_util import load_motion_rep


def get_experiment_dir(result_dir: str, robot: str, task: str):
    exp_name = f"motionbricks_{task}" if robot in ("", "g1") else f"motionbricks_{robot}_{task}"
    return os.path.join(result_dir, exp_name, "version_1")


def stats_folder_from_dataset(dataset_pt: str | None):
    if dataset_pt is None:
        return None
    stats_folder = os.path.join(os.path.dirname(dataset_pt), "stats", "motion")
    return stats_folder if os.path.isdir(stats_folder) else None


def materialize_stats_folder(dataset_pt: str | None, version_dir: str):
    version_stats = os.path.join(version_dir, "stats", "motion")
    dataset_stats = stats_folder_from_dataset(dataset_pt)
    if dataset_stats is not None:
        os.makedirs(version_stats, exist_ok=True)
        for filename in ["mean.npy", "std.npy"]:
            shutil.copy2(os.path.join(dataset_stats, filename), os.path.join(version_stats, filename))
    return version_stats


def load_config(result_dir: str, robot: str, max_steps: int, dataset_pt: str | None):
    """Load and patch hparams.yaml for single-GPU training."""
    version_dir = get_experiment_dir(result_dir, robot, "root")
    hparams_path = os.path.join(version_dir, "hparams.yaml")
    conf = OmegaConf.load(hparams_path)

    stats_folder = materialize_stats_folder(dataset_pt, version_dir)

    with open_dict(conf):
        # resolve data paths to the version directory (where skeleton/stats live)
        conf.data = {"folder": version_dir, "text_embeddings": None}
        conf.skeleton.folder = os.path.join(version_dir, "skeleton")
        conf.motion_rep.stats.folder = stats_folder

        # single-GPU training overrides
        conf.trainer.devices = 1
        conf.trainer.num_nodes = 1
        conf.trainer.max_steps = max_steps
        conf.trainer.accelerator = "auto"
        conf.trainer.strategy = "auto"
        conf.trainer.enable_progress_bar = True
        conf.trainer.log_every_n_steps = 10
        conf.trainer.val_check_interval = max_steps
        conf.trainer.num_sanity_val_steps = 0

        # resolve ${trainer.max_steps} in scheduler
        conf.model.scheduler.num_training_steps = max_steps

        # remove keys with unresolvable ${hydra:...} interpolations
        conf.id = "synthetic"
        conf.run_dir = "."
        conf.out_dir = result_dir

    # resolve all ${} interpolations, then re-wrap as DictConfig
    resolved = OmegaConf.to_container(conf, resolve=True)
    conf = OmegaConf.create(resolved)

    return conf, version_dir


def main():
    parser = argparse.ArgumentParser(description="Root model training")
    parser.add_argument("--result_dir", type=str, default="./out",
                        help="Directory containing pretrained checkpoints")
    parser.add_argument("--robot", type=str, default="g1",
                        help="Robot experiment suffix, e.g. 'vr_h3' uses out/motionbricks_vr_h3_root")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Number of training steps. Defaults to 50000 with --dataset_pt, otherwise 200.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of synthetic samples in dataset")
    parser.add_argument("--dataset_pt", type=str, default=None,
                        help="Optional precomputed motion feature dataset from scripts/prepare_vr_h3_data.py")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_steps is None:
        args.max_steps = 50000 if args.dataset_pt is not None else 200

    pl.seed_everything(args.seed)
    conf, version_dir = load_config(args.result_dir, args.robot, args.max_steps, args.dataset_pt)

    # instantiate skeleton and motion representation
    motion_rep = load_motion_rep(conf)
    feat_dim = len(motion_rep.indices['all'])

    if args.dataset_pt is None:
        dataset = SyntheticMotionDataset(
            feat_dim=feat_dim,
            num_samples=args.num_samples,
            min_frames=200,
            max_frames=400,
        )
    else:
        dataset = TorchMotionDataset(args.dataset_pt)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        persistent_workers=args.num_workers > 0,
    )

    # instantiate networks and model
    model_conf = copy.deepcopy(conf.model)
    with open_dict(model_conf):
        # instantiate backbone network (needs full motion_rep for dual_rep access)
        backbone_net = instantiate(
            model_conf.backbone_network,
            motion_rep=motion_rep,
            _recursive_=False,
        )

        # build optimizer and scheduler as partials
        optimizer_fn = instantiate(model_conf.optimizer)
        scheduler_fn = instantiate(model_conf.scheduler) if model_conf.scheduler else None

        model = instantiate(
            model_conf,
            pose_vqvae_network=None,
            root_vqvae_network=None,
            backbone_network=backbone_net,
            motion_rep=motion_rep,
            optimizer=optimizer_fn,
            scheduler=scheduler_fn,
            _recursive_=False,
        )

    checkpoint_dir = os.path.join(version_dir, "checkpoints")
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="model-{step:07d}",
        every_n_train_steps=args.max_steps,
        save_top_k=-1,
        save_last=True,
    )
    metrics_logger = pl.loggers.CSVLogger(
        save_dir=version_dir,
        name="training_logs",
        version="latest",
    )

    trainer = pl.Trainer(
        max_steps=conf.trainer.max_steps,
        devices=conf.trainer.devices,
        num_nodes=conf.trainer.num_nodes,
        accelerator=conf.trainer.accelerator,
        strategy=conf.trainer.strategy,
        precision=conf.trainer.precision,
        gradient_clip_val=conf.trainer.gradient_clip_val,
        enable_progress_bar=conf.trainer.enable_progress_bar,
        log_every_n_steps=conf.trainer.log_every_n_steps,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
        callbacks=[checkpoint_cb],
        logger=metrics_logger,
    )

    print(f"Starting root model training for {args.max_steps} steps...")
    print(f"  Robot: {args.robot}")
    print(f"  Version dir: {version_dir}")
    print(f"  Feature dim: {feat_dim}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Dataset pt: {args.dataset_pt}")
    print(f"  DataLoader workers: {args.num_workers}")
    trainer.fit(model, train_dataloaders=dataloader)
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Metrics: {metrics_logger.log_dir}/metrics.csv")
    print("Training complete.")


if __name__ == "__main__":
    main()
