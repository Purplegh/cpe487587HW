

from __future__ import annotations

import argparse
import os
import glob
import numpy as np
import torch
import onnxruntime as ort
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision.utils import make_grid

from cpe487587hw import deepl





def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HW04 — Inference + Evaluation")

    p.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory where ONNX files were saved by genmodel_impl.py",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Directory to save generated images and plots",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=25,
        help="Number of images to generate per model (default: 25)",
    )
    p.add_argument(
        "--latent_dim",
        type=int,
        default=256,
        help="Latent dimension used during training (default: 256)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cuda | cpu",
    )

    return p.parse_args()




def _find_latest_onnx(save_dir: str, model_type: str) -> str:
    
    # prefer _final, else pick the highest epoch number
    final = os.path.join(save_dir, f"{model_type}_*_final.onnx")
    finals = glob.glob(final)
    if finals:
        return sorted(finals)[-1]

    pattern = os.path.join(save_dir, f"{model_type}_epoch*.onnx")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(
            f"No ONNX file found for '{model_type}' in '{save_dir}'.\n"
            f"Run genmodel_impl.py --model {model_type} first."
        )
    return sorted(candidates)[-1]


def _load_session(onnx_path: str) -> ort.InferenceSession:
    
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ort.get_device() == "GPU"
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(onnx_path, providers=providers)
    print(f"  Loaded : {onnx_path}")
    return sess



# Image generators  


def generate_vae(
    sess:       ort.InferenceSession,
    n:          int,
    latent_dim: int,
) -> torch.Tensor:
   
    z      = np.random.randn(n, latent_dim).astype(np.float32)
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    images = sess.run([output_name], {input_name: z})[0]   # (n, 3, 64, 64)
    return torch.from_numpy(images).clamp(-1, 1)


def generate_gan(
    sess:       ort.InferenceSession,
    n:          int,
    latent_dim: int,
) -> torch.Tensor:
    
    z      = np.random.randn(n, latent_dim).astype(np.float32)
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    images = sess.run([output_name], {input_name: z})[0]
    return torch.from_numpy(images).clamp(-1, 1)


def generate_diffusion(sess, n, T=1000):
    betas     = np.linspace(1e-4, 2e-2, T).astype(np.float32)
    alphas    = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    alpha_bar_prev = np.concatenate([[1.0], alpha_bar[:-1]])
    posterior_var  = betas * (1 - alpha_bar_prev) / (1 - alpha_bar)

    in_name_x  = sess.get_inputs()[0].name
    in_name_t  = sess.get_inputs()[1].name
    out_name   = sess.get_outputs()[0].name

    all_images = []
    for sample_idx in range(n):
        print(f"  Generating sample {sample_idx+1}/{n} ...")
        x = np.random.randn(1, 3, 64, 64).astype(np.float32)  # batch=1

        for t_idx in reversed(range(T)):
           
            t_batch  = np.array([t_idx], dtype=np.int64)       # shape (1,)
            eps_pred = sess.run([out_name], {in_name_x: x, in_name_t: t_batch})[0]

            beta_t      = betas[t_idx]
            alpha_t     = alphas[t_idx]
            alpha_bar_t = alpha_bar[t_idx]

            coeff = np.float32(beta_t / np.sqrt(1 - alpha_bar_t))
            mean  = (np.float32(1 / np.sqrt(alpha_t)) * (x - coeff * eps_pred)).astype(np.float32)

            if t_idx > 0:
                noise = np.random.randn(*x.shape).astype(np.float32)
                x     = (mean + np.float32(np.sqrt(posterior_var[t_idx])) * noise).astype(np.float32)
            else:
                x = mean.astype(np.float32)

                all_images.append(x)  # (1, 3, 64, 64)

    images = np.concatenate(all_images, axis=0)  # (n, 3, 64, 64)
    return torch.from_numpy(images).clamp(-1, 1)




def _denorm(images: torch.Tensor) -> torch.Tensor:
    
    return (images + 1.0) / 2.0


def save_image_grid(images: torch.Tensor, path: str, title: str):
    
    grid = make_grid(_denorm(images[:25]), nrow=5, padding=2)
    npimg = grid.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(npimg)
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved image grid : {path}")


def plot_comparison(
    all_metrics: dict[str, dict[str, list[float]]],
    out_dir:     str,
):
    
    metric_names = ["VoL", "TEN", "HFE", "MLSD", "FCON"]
    model_names  = list(all_metrics.keys())
    colors       = ["#4C72B0", "#DD8452", "#55A868"]   # blue, orange, green

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "Image Quality Metrics: VAE vs GAN vs Diffusion Model",
        fontsize=14, fontweight="bold", y=1.01,
    )

    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.35)

    for col, metric in enumerate(metric_names):
        ax = fig.add_subplot(gs[col])

        # Collect data per model for this metric
        data   = [all_metrics[m][metric] for m in model_names]
        pos    = np.arange(len(model_names))

        # Box plot
        bp = ax.boxplot(
            data,
            positions=pos,
            widths=0.5,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Overlay individual points
        for i, (d, color) in enumerate(zip(data, colors)):
            jitter = np.random.uniform(-0.15, 0.15, len(d))
            ax.scatter(
                np.full(len(d), i) + jitter, d,
                color=color, alpha=0.5, s=15, zorder=3,
            )

        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xticks(pos)
        ax.set_xticklabels(model_names, fontsize=10, rotation=15)
        ax.set_ylabel("Score", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    path = os.path.join(out_dir, "metrics_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved comparison plot : {path}")


def save_metrics_csv(
    all_metrics: dict[str, dict[str, list[float]]],
    out_dir:     str,
):
   
    import csv
    path = os.path.join(out_dir, "metrics_raw.csv")
    metric_names = ["VoL", "TEN", "HFE", "MLSD", "FCON"]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "image_idx"] + metric_names)
        for model_name, metrics in all_metrics.items():
            n = len(metrics["VoL"])
            for i in range(n):
                row = [model_name, i] + [metrics[m][i] for m in metric_names]
                writer.writerow(row)

    print(f"  Saved raw metrics CSV : {path}")



# Main


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("  HW04 — Inference + Evaluation")
    print("=" * 60)
    print(f"ONNX dir  : {args.save_dir}")
    print(f"Output dir: {args.out_dir}")
    print(f"Samples   : {args.n_samples} per model")
    print("=" * 60)

    model_types = ["vae", "gan", "diffusion"]
    all_metrics: dict[str, dict[str, list[float]]] = {}

    for model_type in model_types:
        print(f"\n {model_type.upper()}")

        # -- Load ONNX --
        try:
            onnx_path = _find_latest_onnx(args.save_dir, model_type)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            print(f"  Skipping {model_type.upper()}")
            continue

        sess = _load_session(onnx_path)

        # -- Generate 25 images --
        print(f"  Generating {args.n_samples} images ...")
        if model_type == "vae":
            images = generate_vae(sess, args.n_samples, args.latent_dim)
        elif model_type == "gan":
            images = generate_gan(sess, args.n_samples, args.latent_dim)
        else:
            images = generate_diffusion(sess, args.n_samples)

        print(f"  Generated shape: {images.shape}")   # (25, 3, 64, 64)

        # -- Save image grid --
        grid_path = os.path.join(args.out_dir, f"{model_type}_samples.png")
        save_image_grid(images, grid_path, title=f"{model_type.upper()} — 25 Generated Samples")

        # -- Compute metrics --
        print(f"  Computing metrics ...")
        metrics = deepl.compute_metrics_batch(images)
        all_metrics[model_type.upper()] = metrics

        # Print summary
        print(f"  Metric summary (mean ± std):")
        for name, vals in metrics.items():
            arr = np.array(vals)
            print(f"    {name:5s} : {arr.mean():.4f} ± {arr.std():.4f}")

    # -- Comparison plot --
    if len(all_metrics) > 0:
        print("\nGenerating comparison plot ...")
        plot_comparison(all_metrics, args.out_dir)
        save_metrics_csv(all_metrics, args.out_dir)
    else:
        print("\nNo models evaluated — train at least one model first!")

    print("\nDone!")


if __name__ == "__main__":
    main()