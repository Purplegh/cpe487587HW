

from __future__ import annotations

import argparse
import os
import zipfile
import io

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from cpe487587hw import deepl





class CelebAZipDataset(Dataset):
   

    def __init__(self, zip_path: str, transform=None):
        self.zip_path  = zip_path
        self.transform = transform

        with zipfile.ZipFile(zip_path, 'r') as zf:
            self.image_names = sorted([
                name for name in zf.namelist()
                if name.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            with zf.open(self.image_names[idx]) as f:
                img = Image.open(io.BytesIO(f.read())).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img




def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HW04 — Train a generative model on CelebA")

    p.add_argument(
        "--model",
        choices=["vae", "gan", "diffusion"],
        required=True,
        help="Which model to train: vae | gan | diffusion",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    p.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Fraction of selected data to use for training (default: 0.9 = 90%% train, 10%% val)",
    )
    p.add_argument(
        "--data_ratio",
        type=float,
        default=1.0,
        help="Fraction of the full CelebA dataset to use (default: 1.0 = 100%%). "
             "Use 0.02 for a quick 2%% test run.",
    )
    p.add_argument(
        "--onnx_every",
        type=int,
        default=10,
        help="Save an ONNX checkpoint every this many epochs (default: 10)",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training (default: 128)",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    p.add_argument(
        "--zip_path",
        type=str,
        default="/data/CPE_487-587/img_align_celeba.zip",
        help="Path to CelebA zip file on Lovelace",
    )
    p.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save ONNX checkpoints (default: checkpoints/)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on: cuda | cpu",
    )
    p.add_argument(
        "--latent_dim",
        type=int,
        default=256,
        help="Latent dimension for VAE or GAN (default: 256)",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4)",
    )

    return p.parse_args()




def build_dataloaders(
    zip_path:    str,
    data_ratio:  float,
    train_ratio: float,
    batch_size:  int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
   
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1]
    ])

    full_dataset = CelebAZipDataset(zip_path, transform=transform)
    n_full       = len(full_dataset)

    # Step 1: subset the dataset by data_ratio --
    n_use    = max(1, int(n_full * data_ratio))
    n_discard = n_full - n_use
    subset, _ = random_split(
        full_dataset, [n_use, n_discard],
        generator=torch.Generator().manual_seed(42),
    )

    # Step 2: split subset into train / val by train_ratio 
    n_train = int(n_use * train_ratio)
    n_val   = n_use - n_train

    train_ds, val_ds = random_split(
        subset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"Full dataset  : {n_full:,} images")
    print(f"Using         : {n_use:,} images  ({data_ratio*100:.1f}% of full)")
    print(f"Train split   : {n_train:,} images  ({train_ratio*100:.0f}%)")
    print(f"Val split     : {n_val:,} images  ({(1-train_ratio)*100:.0f}%)")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader





def build_model(model_type: str, latent_dim: int) -> torch.nn.Module:
    if model_type == "vae":
        model = deepl.VAE(in_channels=3, latent_dim=latent_dim)
    elif model_type == "gan":
        model = deepl.GAN(in_channels=3, latent_dim=latent_dim)
    elif model_type == "diffusion":
        model = deepl.DiffusionModel(in_channels=3, T=1000)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model         : {model_type.upper()}")
    print(f"Parameters    : {n_params:,}")
    return model



# Main


def main():
    args   = parse_args()
    device = torch.device(args.device)

   
    print(f"  HW04 — Generative Model Training")
    print("=" * 60)
    print(f"Model      : {args.model.upper()}")
    print(f"Epochs     : {args.epochs}")
    print(f"Data ratio : {args.data_ratio} ({args.data_ratio*100:.1f}% of CelebA)")
    print(f"Train ratio: {args.train_ratio} ({args.train_ratio*100:.0f}% train / {(1-args.train_ratio)*100:.0f}% val)")
    print(f"Batch size : {args.batch_size}")
    print(f"LR         : {args.lr}")
    print(f"ONNX every : {args.onnx_every} epochs")
    print(f"Device     : {device}")
    print(f"Save dir   : {args.save_dir}")
  

    # Dataset 
    train_loader, val_loader = build_dataloaders(
        zip_path    = args.zip_path,
        data_ratio  = args.data_ratio,
        train_ratio = args.train_ratio,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )

 
    model = build_model(args.model, args.latent_dim)

    
    trainer = deepl.GenModelTrainer(
        model      = model,
        model_type = args.model,
        device     = device,
        lr         = args.lr,
        save_dir   = args.save_dir,
        onnx_every = args.onnx_every,
    )

    
    print(f"\nStarting training ...\n")
    trainer.train(loader=train_loader, epochs=args.epochs)

    print("\nTraining complete!")
    print(f"ONNX files saved in: {os.path.abspath(args.save_dir)}/")


if __name__ == "__main__":
    main()