
import argparse
import sys

import numpy as np
import pandas as pd
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser(
        description='ACC state inference using trained ONNX model.'
    )
    parser.add_argument('--speed_csv',   required=True,
                        help='Path to decoded_wheel_speed_fl.csv for unseen experiment.')
    parser.add_argument('--acc_csv',     required=False, default=None,
                        help='Path to decoded_acc_status.csv for accuracy evaluation.')
    parser.add_argument('--onnx_model',  required=True,
                        help='Path to acc_model.onnx.')
    parser.add_argument('--norm_coeffs', required=True,
                        help='Path to acc_norm_coeffs.npz.')
    parser.add_argument('--k', type=int, default=10,
                        help='History length used during training (default 10).')
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load normalisation coefficients
    coeffs    = np.load(args.norm_coeffs)
    norm_mean = float(coeffs['mean'])
    norm_std  = float(coeffs['std'])
    print(f"Normalisation - mean: {norm_mean:.4f} m/s, std: {norm_std:.4f} m/s")

    # 2. Load and preprocess speed CSV
    df = pd.read_csv(args.speed_csv, usecols=['Time', 'Message'])
    df.rename(columns={'Message': 'speed_kmh'}, inplace=True)
    df['speed_ms'] = df['speed_kmh'] / 3.6
    df.sort_values('Time', inplace=True)
    df.reset_index(drop=True, inplace=True)

    speeds = df['speed_ms'].values
    speeds = (speeds - norm_mean) / (norm_std + 1e-8)

    n = len(speeds)
    if n <= args.k:
        print(f"ERROR: Only {n} speed samples, need more than k={args.k}.")
        sys.exit(1)

    # 3. Derive feature groups from speed signal
    accel       = np.diff(speeds, prepend=speeds[0]).astype(np.float32)
    jerk        = np.diff(accel,  prepend=accel[0]).astype(np.float32)
    rolling_std = np.array([
        speeds[max(0, i - args.k): i + 1].std()
        for i in range(n)
    ], dtype=np.float32)

    # 4. Build sliding window features (4*(k+1) = 44 per sample)
    X_rows = []
    for t in range(args.k, n):
        s  = speeds[t - args.k: t + 1][::-1].copy()
        a  = accel[t - args.k: t + 1][::-1].copy()
        j  = jerk[t - args.k: t + 1][::-1].copy()
        rs = rolling_std[t - args.k: t + 1][::-1].copy()
        X_rows.append(np.concatenate([s, a, j, rs]))

    X = np.array(X_rows, dtype=np.float32)   # (N, 44)

    # 5. ONNX inference
    session    = ort.InferenceSession(args.onnx_model)
    input_name = session.get_inputs()[0].name
    logits     = session.run(None, {input_name: X})[0]
    probs      = 1.0 / (1.0 + np.exp(-logits))
    preds      = (probs >= 0.5).astype(int)

    # 6. Report prediction distribution
    n_acc     = preds.sum()
    n_samples = len(preds)
    print(f"\nProcessed {n_samples} windows.")
    print(f"  ACC enabled (1): {n_acc}  ({n_acc/n_samples*100:.1f}%)")
    print(f"  ACC off     (0): {n_samples - n_acc}  ({(n_samples-n_acc)/n_samples*100:.1f}%)")

    # 7. Compute accuracy if ground truth ACC CSV is provided
    if args.acc_csv is not None:
        acc_df = pd.read_csv(args.acc_csv, usecols=['Time', 'Message'])
        acc_df.rename(columns={'Message': 'acc_status'}, inplace=True)
        acc_df.sort_values('Time', inplace=True)
        acc_df.reset_index(drop=True, inplace=True)
        acc_df['label'] = (acc_df['acc_status'] == 6).astype(int)

        # ZOH: align ACC labels to speed timestamps
        speed_times = df.iloc[args.k:].reset_index(drop=True)
        merged = pd.merge_asof(
            speed_times[['Time']],
            acc_df[['Time', 'label']],
            on='Time',
            direction='backward'
        )
        merged.dropna(subset=['label'], inplace=True)
        labels        = merged['label'].values[:len(preds)].astype(int)
        preds_trimmed = preds[:len(labels)]

        correct = (preds_trimmed == labels).sum()
        total   = len(labels)
        acc     = correct / total * 100
        print(f"\nAccuracy: {acc:.2f}%  ({correct}/{total} correct)")


if __name__ == '__main__':
    main()