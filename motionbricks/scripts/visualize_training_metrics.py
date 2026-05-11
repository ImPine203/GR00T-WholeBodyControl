"""Create plots and a compact summary from MotionBricks CSV training logs."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RUNS = {
    "vqvae": "out/motionbricks_vr_h3_vqvae/version_1/training_logs/latest/metrics.csv",
    "pose": "out/motionbricks_vr_h3_pose/version_1/training_logs/latest/metrics.csv",
    "root": "out/motionbricks_vr_h3_root/version_1/training_logs/latest/metrics.csv",
}

PLOT_GROUPS = {
    "vqvae": [
        ("vqvae_loss", ["loss/train_loss_step", "loss/train_l_recons_pose_step"]),
        ("vqvae_reconstruction", ["loss/train_l_recons_pose_step", "loss/train_l_joint_vel_step"]),
        ("vqvae_codebook", ["loss/train_perplexity_pose_step", "loss/train_l_commit_pose_step"]),
    ],
    "pose": [
        ("pose_loss", ["loss/train_loss_step", "loss/train_pose_loss_step"]),
    ],
    "root": [
        ("root_loss", ["loss/train_loss_step", "loss/train_num_token_loss_step"]),
        ("root_reconstruction", ["loss/train_global_root_recons_loss_step", "loss/train_local_root_recons_loss_step"]),
        ("root_accuracy", ["loss/train_top_1_accuracy_step", "loss/train_top_3_accuracy_step", "loss/train_top_5_accuracy_step"]),
    ],
}


def read_metrics(path: Path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")

    series = {}
    for column in rows[0].keys():
        values = []
        steps = []
        for row in rows:
            value = row.get(column, "")
            step = row.get("step", "")
            if value == "" or step == "":
                continue
            try:
                values.append(float(value))
                steps.append(float(step))
            except ValueError:
                continue
        if values:
            series[column] = (steps, values)
    return series


def moving_average(values, window):
    if window <= 1 or len(values) <= window:
        return values
    out = []
    running = 0.0
    for idx, value in enumerate(values):
        running += value
        if idx >= window:
            running -= values[idx - window]
        out.append(running / min(idx + 1, window))
    return out


def summarize_metric(values):
    tail = values[-min(50, len(values)):]
    return {
        "first": values[0],
        "last": values[-1],
        "min": min(values),
        "max": max(values),
        "tail_mean": sum(tail) / len(tail),
        "n": len(values),
    }


def plot_group(name, group_name, columns, series, output_dir, smooth_window):
    present = [column for column in columns if column in series]
    if not present:
        return None

    plt.figure(figsize=(10, 5))
    for column in present:
        steps, values = series[column]
        plt.plot(steps, moving_average(values, smooth_window), label=column.replace("loss/train_", ""))
    plt.xlabel("step")
    plt.ylabel("value")
    plt.title(f"{name}: {group_name}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path = output_dir / f"{name}_{group_name}.png"
    plt.savefig(output_path, dpi=140)
    plt.close()
    return output_path


def write_summary(all_series, output_dir):
    summary_path = output_dir / "summary.md"
    with summary_path.open("w") as f:
        f.write("# Training Metrics Summary\n\n")
        for name, series in all_series.items():
            f.write(f"## {name}\n\n")
            for column in sorted(series):
                if column in ("epoch", "step") or not column.startswith("loss/"):
                    continue
                _, values = series[column]
                stat = summarize_metric(values)
                f.write(
                    f"- `{column}`: first={stat['first']:.6g}, "
                    f"last={stat['last']:.6g}, min={stat['min']:.6g}, "
                    f"max={stat['max']:.6g}, tail_mean={stat['tail_mean']:.6g}, n={stat['n']}\n"
                )
            f.write("\n")
    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Visualize MotionBricks CSV training metrics")
    parser.add_argument("--output_dir", type=str, default="out/vr_h3_training_plots")
    parser.add_argument("--smooth_window", type=int, default=20)
    parser.add_argument("--vqvae_csv", type=str, default=DEFAULT_RUNS["vqvae"])
    parser.add_argument("--pose_csv", type=str, default=DEFAULT_RUNS["pose"])
    parser.add_argument("--root_csv", type=str, default=DEFAULT_RUNS["root"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = {
        "vqvae": Path(args.vqvae_csv),
        "pose": Path(args.pose_csv),
        "root": Path(args.root_csv),
    }

    all_series = {}
    generated = []
    for name, path in csv_paths.items():
        if not path.exists():
            print(f"Skip {name}: metrics file not found: {path}")
            continue
        series = read_metrics(path)
        all_series[name] = series
        for group_name, columns in PLOT_GROUPS.get(name, []):
            output_path = plot_group(name, group_name, columns, series, output_dir, args.smooth_window)
            if output_path is not None:
                generated.append(output_path)

    if not all_series:
        raise FileNotFoundError("No metrics CSV files found")

    summary_path = write_summary(all_series, output_dir)
    print(f"Summary: {summary_path}")
    for path in generated:
        print(f"Plot: {path}")


if __name__ == "__main__":
    main()
