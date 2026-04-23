

from __future__ import annotations

import os
import math
import argparse
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import onnx  


#  sinusoidal time-step embedding 


class SinusoidalEmbedding(nn.Module):
   

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) integer timesteps
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]          # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1) # (B, dim)
        return emb



# 1.  VAE  –  Variational AutoEncoder


class VAEEncoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        # 64x64 -> 32 -> 16 -> 8 -> 4
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64,  4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64,          128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128,         256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256,         512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
        )
        self.fc_mu     = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.net(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, z: torch.Tensor):
        h = self.fc(z).view(-1, 512, 4, 4)
        return self.net(h)


class VAE(nn.Module):
    

    def __init__(self, in_channels: int = 3, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(in_channels, latent_dim)
        self.decoder = VAEDecoder(latent_dim, in_channels)

   
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
       
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    
    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

  
    @staticmethod
    def loss(recon: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        ELBO loss = reconstruction (MSE) + KL divergence.
        Both terms are averaged over the batch.
        """
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        # KL: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample n images from the prior N(0, I)."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)



# 2.  GAN  –  Generative Adversarial Network


class GANGenerator(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 512, 4, 4)
        return self.net(h)


class GANDiscriminator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64,  4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64,          128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128,         256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256,         512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512,         1,   4, 1, 0),                       # (B,1,1,1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)   # (B,)


class GAN(nn.Module):
    

    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.latent_dim    = latent_dim
        self.generator     = GANGenerator(latent_dim, in_channels)
        self.discriminator = GANDiscriminator(in_channels)


    def forward(self, z: torch.Tensor) -> torch.Tensor:
       
        return self.generator(z)

    
    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
       
        z = torch.randn(n, self.latent_dim, device=device)
        return self.generator(z)



# 3.  DiffusionModel 


# UNet building blocks 

class ResBlock(nn.Module):

    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.res   = ResBlock(in_ch, time_dim)
        self.down  = nn.Conv2d(in_ch, out_ch, 4, 2, 1)

    def forward(self, x, t_emb):
        x = self.res(x, t_emb)
        return self.down(x), x          # return (downsampled, skip)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.res  = ResBlock(out_ch * 2, time_dim)  # *2 for skip concat
        self.proj = nn.Conv2d(out_ch * 2, out_ch, 1) # merge back to out_ch

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t_emb)
        return self.proj(x)


class UNet(nn.Module):
   

    def __init__(self, in_channels: int = 3, base_ch: int = 64, time_dim: int = 256):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        td = time_dim

        self.init_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)

        self.down1 = DownBlock(base_ch,      base_ch * 2, td)  # 64->32
        self.down2 = DownBlock(base_ch * 2,  base_ch * 4, td)  # 32->16
        self.down3 = DownBlock(base_ch * 4,  base_ch * 8, td)  # 16->8

        self.mid_res = ResBlock(base_ch * 8, td)

        self.up3 = UpBlock(base_ch * 8, base_ch * 4, td)   # 8->16
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, td)   # 16->32
        self.up1 = UpBlock(base_ch * 2, base_ch,     td)   # 32->64

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, in_channels, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)                     # (B, time_dim)

        x = self.init_conv(x)                         # (B, C, 64, 64)

        x, s1 = self.down1(x, t_emb)                 # (B,2C,32,32)
        x, s2 = self.down2(x, t_emb)                 # (B,4C,16,16)
        x, s3 = self.down3(x, t_emb)                 # (B,8C, 8, 8)

        x = self.mid_res(x, t_emb)

        x = self.up3(x, s3, t_emb)                   # (B,4C,16,16)
        x = self.up2(x, s2, t_emb)                   # (B,2C,32,32)
        x = self.up1(x, s1, t_emb)                   # (B, C,64,64)

        return self.out_conv(x)                       # (B, in_ch, 64, 64)


class DiffusionModel(nn.Module):
   

    def __init__(
        self,
        in_channels: int = 3,
        T:           int = 1000,
        beta_start: float = 1e-4,
        beta_end:   float = 2e-2,
    ):
        super().__init__()
        self.T          = T
        self.in_channels = in_channels
        self.unet       = UNet(in_channels)

     
        betas              = torch.linspace(beta_start, beta_end, T)
        alphas             = 1.0 - betas
        alpha_bar          = torch.cumprod(alphas, dim=0)
        alpha_bar_prev     = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

        self.register_buffer('betas',          betas)
        self.register_buffer('alphas',         alphas)
        self.register_buffer('alpha_bar',      alpha_bar)
        self.register_buffer('alpha_bar_prev', alpha_bar_prev)
        self.register_buffer('sqrt_alpha_bar',      alpha_bar.sqrt())
        self.register_buffer('sqrt_one_minus_ab',   (1 - alpha_bar).sqrt())
        self.register_buffer('posterior_var',
            betas * (1 - alpha_bar_prev) / (1 - alpha_bar))


    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor | None = None):
        """Forward process: add noise to x0 at timestep t."""
        if eps is None:
            eps = torch.randn_like(x0)
        s1 = self.sqrt_alpha_bar[t][:, None, None, None]
        s2 = self.sqrt_one_minus_ab[t][:, None, None, None]
        return s1 * x0 + s2 * eps, eps


    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Training forward: sample random t, add noise, predict noise.
        Returns the simple MSE loss ||eps - eps_theta||^2.
        """
        B      = x0.shape[0]
        device = x0.device
        t      = torch.randint(0, self.T, (B,), device=device)
        xt, eps = self.q_sample(x0, t)
        eps_pred = self.unet(xt, t)
        return F.mse_loss(eps_pred, eps)

   
    @torch.no_grad()
    def sample(self, n: int, device: torch.device, image_size: int = 64) -> torch.Tensor:
        
        x = torch.randn(n, self.in_channels, image_size, image_size, device=device)
        for t_idx in reversed(range(self.T)):
            t_batch = torch.full((n,), t_idx, device=device, dtype=torch.long)
            eps_pred = self.unet(x, t_batch)

            beta_t     = self.betas[t_idx]
            alpha_t    = self.alphas[t_idx]
            alpha_bar_t = self.alpha_bar[t_idx]

            # DDPM reverse mean
            coeff = beta_t / (1 - alpha_bar_t).sqrt()
            mean  = (1 / alpha_t.sqrt()) * (x - coeff * eps_pred)

            if t_idx > 0:
                noise = torch.randn_like(x)
                var   = self.posterior_var[t_idx]
                x     = mean + var.sqrt() * noise
            else:
                x = mean

        return x.clamp(-1, 1)



# 4.  GenModelTrainer  –  Unified trainer for all three models


ModelType = Literal["vae", "gan", "diffusion"]


class GenModelTrainer:
   

    def __init__(
        self,
        model:      nn.Module,
        model_type: ModelType,
        device:     torch.device,
        lr:         float = 2e-4,
        save_dir:   str   = "checkpoints",
        onnx_every: int   = 5,
    ):
        self.model      = model.to(device)
        self.model_type = model_type.lower()
        self.device     = device
        self.save_dir   = save_dir
        self.onnx_every = onnx_every
        os.makedirs(save_dir, exist_ok=True)

        # Optimizers
        if self.model_type == "gan":
            self.opt_g = torch.optim.Adam(model.generator.parameters(),     lr=lr, betas=(0.5, 0.999))
            self.opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        else:
            self.opt = torch.optim.Adam(model.parameters(), lr=lr)

    
    def _train_one_epoch_vae(self, loader: DataLoader) -> float:
        self.model.train()
        total = 0.0
        for batch in loader:
            x = batch.to(self.device)
            self.opt.zero_grad()
            recon, mu, logvar = self.model(x)
            loss = VAE.loss(recon, x, mu, logvar)
            loss.backward()
            self.opt.step()
            total += loss.item()
        return total / len(loader)

   
    def _train_one_epoch_gan(self, loader: DataLoader) -> float:
        self.model.train()
        bce      = nn.BCEWithLogitsLoss()
        g_losses = []
        d_losses = []

        for batch in loader:
            real = batch.to(self.device)
            B    = real.size(0)

            #  Train Discriminator 
            self.opt_d.zero_grad()
            real_logits = self.model.discriminator(real)
            d_real_loss = bce(real_logits, torch.ones(B, device=self.device))

            z            = torch.randn(B, self.model.latent_dim, device=self.device)
            fake         = self.model.generator(z).detach()
            fake_logits  = self.model.discriminator(fake)
            d_fake_loss  = bce(fake_logits, torch.zeros(B, device=self.device))

            d_loss = (d_real_loss + d_fake_loss) * 0.5
            d_loss.backward()
            self.opt_d.step()

            #Train Generator 
            self.opt_g.zero_grad()
            z           = torch.randn(B, self.model.latent_dim, device=self.device)
            fake        = self.model.generator(z)
            fake_logits = self.model.discriminator(fake)
            g_loss      = bce(fake_logits, torch.ones(B, device=self.device))
            g_loss.backward()
            self.opt_g.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        avg_g = sum(g_losses) / len(g_losses)
        avg_d = sum(d_losses) / len(d_losses)
        return (avg_g + avg_d) / 2

   
    def _train_one_epoch_diffusion(self, loader: DataLoader) -> float:
        self.model.train()
        total = 0.0
        for batch in loader:
            x = batch.to(self.device)
            self.opt.zero_grad()
            loss = self.model(x)          # forward() returns the MSE loss
            loss.backward()
            self.opt.step()
            total += loss.item()
        return total / len(loader)

    def train(self, loader: DataLoader, epochs: int):
       
        dispatch = {
            "vae":       self._train_one_epoch_vae,
            "gan":       self._train_one_epoch_gan,
            "diffusion": self._train_one_epoch_diffusion,
        }
        train_fn = dispatch[self.model_type]

        for epoch in range(1, epochs + 1):
            avg_loss = train_fn(loader)
            print(f"[{self.model_type.upper()}] Epoch {epoch:04d}/{epochs}  loss={avg_loss:.4f}")

            if epoch % self.onnx_every == 0:
                self.save_onnx(epoch)

        # Always save at the end
        self.save_onnx(epoch=epochs, suffix="_final")

    def save_onnx(self, epoch: int, suffix: str = ""):
      
        path = os.path.join(
            self.save_dir, f"{self.model_type}_epoch{epoch:04d}{suffix}.onnx"
        )
        self.model.eval()

        with torch.no_grad():
            if self.model_type == "vae":
                # Export decoder: input = latent z
                dummy = torch.randn(1, self.model.latent_dim, device=self.device)
                torch.onnx.export(
                    self.model.decoder, dummy, path,
                    input_names=["z"], output_names=["image"],
                    dynamic_axes={"z": {0: "batch"}, "image": {0: "batch"}},
                    opset_version=17,
                )

            elif self.model_type == "gan":
                # Export generator: input = noise z
                dummy = torch.randn(1, self.model.latent_dim, device=self.device)
                torch.onnx.export(
                    self.model.generator, dummy, path,
                    input_names=["z"], output_names=["image"],
                    dynamic_axes={"z": {0: "batch"}, "image": {0: "batch"}},
                    opset_version=17,
                )

            elif self.model_type == "diffusion":
                # Export UNet: inputs = noisy_image, timestep
                dummy_x = torch.randn(1, self.model.in_channels, 64, 64, device=self.device)
                dummy_t = torch.zeros(1, dtype=torch.long, device=self.device)
                torch.onnx.export(
                    self.model.unet, (dummy_x, dummy_t), path,
                    input_names=["noisy_image", "timestep"],
                    output_names=["predicted_noise"],
                    dynamic_axes={
                        "noisy_image":     {0: "batch"},
                        "predicted_noise": {0: "batch"},
                    },
                    opset_version=17,
                )

        print(f"  -> ONNX saved: {path}")
        self.model.train()



# CLI entry point (for quick testing of this file directly)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test gen_model.py instantiation")
    p.add_argument("--model",  choices=["vae", "gan", "diffusion"], default="vae")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    device = torch.device(args.device)

    if args.model == "vae":
        m = VAE()
    elif args.model == "gan":
        m = GAN()
    else:
        m = DiffusionModel()

    m = m.to(device)
    imgs = m.sample(4, device)
    print(f"{args.model.upper()} sample shape: {imgs.shape}")   # (4, 3, 64, 64)