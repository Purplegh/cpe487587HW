import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_from_disk
import subprocess

from cpe487587hw import deepl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 128


def get_best_gpu(strategy: str = "utilization") -> int:
    """Select best GPU by least utilization or most free memory."""
    if not torch.cuda.is_available():
        return -1

    if strategy == "memory":
        free_mem = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.mem_get_info(i)
            free_mem.append(props[0])
        return free_mem.index(max(free_mem))

    elif strategy == "utilization":
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            utilizations = [int(x.strip()) for x in result.stdout.strip().split("\n")]
            return utilizations.index(min(utilizations))
        except Exception:
            return 0

    return 0


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def preprocess_train(examples):
    images = [train_transform(img.convert("RGB")) for img in examples["image"]]
    return {"pixel_values": images, "labels": examples["label"]}


def preprocess_val(examples):
    images = [val_transform(img.convert("RGB")) for img in examples["image"]]
    return {"pixel_values": images, "labels": examples["label"]}


def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}


def save_example_image(dataset, class_names, split_name: str):
    """Save the first image of a dataset split to the scripts folder."""
    example      = dataset[0]
    image        = example["image"]
    label_id     = example["label"]
    full_label   = class_names[label_id]
    primary_name = full_label.split(",")[0].strip()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image)
    ax.set_title(f"[{split_name}] ID {label_id}: {primary_name}", fontsize=10)
    ax.axis("off")
    path = os.path.join(SCRIPT_DIR, f"example_{split_name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved example {split_name} image -> {path}")


def main():
    parser = argparse.ArgumentParser(description="ImageNet CNN Trainer - HW3 Q6")
    parser.add_argument("--epochs",      type=int,   default=10000,
                        help="Number of training epochs (default: 10000)")
    parser.add_argument("--train_ratio", type=float, default=0.10,
                        help="Fraction of training data to use (default: 0.10)")
    parser.add_argument("--val_ratio",   type=float, default=0.05,
                        help="Fraction of validation data to use (default: 0.05)")
    args = parser.parse_args()

    # ── Device
    device_id = get_best_gpu(strategy="memory")
    if device_id >= 0:
        device = torch.device(f"cuda:{device_id}")
        print(f"Selected GPU: {device_id}")
    else:
        device = torch.device("cpu")
        print("No GPU found - using CPU")

    # ── Load dataset
    print("Loading ImageNet-1k dataset ...")
    dataset = load_from_disk("/data/CPE_487-587/imagenet-1k-arrow")

    train_dataset = dataset["train"]
    val_dataset   = dataset["validation"]

    class_names = train_dataset.features["label"].names
    num_classes = len(class_names)
    print(f"Number of classes       : {num_classes}")
    print(f"Original training size  : {len(train_dataset)}")
    print(f"Original validation size: {len(val_dataset)}")

    # ── Subset
    train_size = int(len(train_dataset) * args.train_ratio)
    val_size   = int(len(val_dataset)   * args.val_ratio)
    print(f"Using {train_size} training samples  ({args.train_ratio*100:.0f}%)")
    print(f"Using {val_size} validation samples  ({args.val_ratio*100:.0f}%)")

    raw_train = train_dataset.select(range(train_size))
    raw_val   = val_dataset.select(range(val_size))

    # ── Save example images to scripts folder
    save_example_image(raw_train, class_names, "train")
    save_example_image(raw_val,   class_names, "validation")

    # ── Apply transforms
    train_ds = raw_train.with_transform(preprocess_train)
    val_ds   = raw_val.with_transform(preprocess_val)

    # ── DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    # ── Model
    model = deepl.ImageNetCNN(num_classes=num_classes)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # ── Optimizer, Scheduler, Loss
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.1
    )

    # ── Trainer
    trainer = deepl.CNNTrainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=SCRIPT_DIR,
    )

    # ── Train
    trainer.train(train_loader, val_loader, num_epochs=args.epochs)

    # ── Save ONNX
    trainer.save_onnx(os.path.join(SCRIPT_DIR, "imagenet_cnn.onnx"))

    # ── Save plot
    trainer.plot_history(save_path=os.path.join(SCRIPT_DIR, "training_history.png"))

    print(f"\nAll done! Files saved to: {SCRIPT_DIR}")


if __name__ == "__main__":
    main()