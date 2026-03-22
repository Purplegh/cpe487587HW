

import argparse
import os
import sys

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    sys.exit(
        "onnxruntime is not installed.\n"
        "Install it with:  pip install onnxruntime   (or onnxruntime-gpu)"
    )


def _load_class_names():
    """Try to load class names from the Arrow dataset cache."""
    try:
        from datasets import load_from_disk
        ds = load_from_disk("/data/CPE_487-587/imagenet-1k-arrow")
        names = ds["train"].features["label"].names
        return [n.split(",")[0].strip() for n in names]
    except Exception:
        # Fall back to numeric labels
        return [str(i) for i in range(1000)]



_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def preprocess(image_path: str) -> np.ndarray:
    """Load an image and return a (1, 3, 224, 224) float32 numpy array."""
    img = Image.open(image_path).convert("RGB")

    # Resize shorter side to 256
    w, h = img.size
    scale = 256 / min(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

    # Center-crop to 224×224
    w, h = img.size
    left = (w - 224) // 2
    top  = (h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))

    # To float32 [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0

    # Normalize
    arr = (arr - _MEAN) / _STD            # (224, 224, 3)
    arr = arr.transpose(2, 0, 1)          # (3, 224, 224)
    arr = np.expand_dims(arr, axis=0)     # (1, 3, 224, 224)
    return arr



def run_inference(onnx_path: str, image_path: str):
    """Run ONNX inference and print the predicted class."""
    if not os.path.isfile(onnx_path):
        sys.exit(f"ONNX file not found: {onnx_path}")
    if not os.path.isfile(image_path):
        sys.exit(f"Image file not found: {image_path}")

    print(f"Loading model : {onnx_path}")
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name

    print(f"Preprocessing : {image_path}")
    input_tensor = preprocess(image_path)

    print("Running inference ...")
    logits = session.run(None, {input_name: input_tensor})[0]  # (1, num_classes)
    logits = logits[0]

    # Softmax
    exp_l = np.exp(logits - logits.max())
    probs = exp_l / exp_l.sum()

    class_names = _load_class_names()
    top_idx = probs.argsort()[::-1][0]
    label = class_names[top_idx] if top_idx < len(class_names) else str(top_idx)

    print(f"\nPredicted class: {label}  ({probs[top_idx]*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="ImageNet CNN - ONNX Inference Script (HW3 Q6)"
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to the input image file (JPEG / PNG / etc.)"
    )
    parser.add_argument(
        "--onnx", required=True,
        help="Path to the trained ONNX model file"
    )
    args = parser.parse_args()

    run_inference(
        onnx_path=args.onnx,
        image_path=args.image,
    )


if __name__ == "__main__":
    main()