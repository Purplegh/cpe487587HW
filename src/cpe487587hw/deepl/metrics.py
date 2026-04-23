

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F




def _to_gray_numpy(image: torch.Tensor) -> np.ndarray:
    
    img = image.detach().cpu().float()

    # If in [-1, 1] range, shift to [0, 1]
    if img.min() < 0:
        img = (img + 1.0) / 2.0

    img = img.clamp(0.0, 1.0)

    if img.ndim == 3:

        if img.shape[0] == 3:
            # weights: R=0.2989, G=0.5870, B=0.1140
            weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(3, 1, 1)
            img = (img * weights).sum(dim=0)
        else:
            img = img.mean(dim=0)

    return img.numpy()   # (H, W) in [0, 1]



def variance_of_laplacian(image: torch.Tensor) -> float:
 
    gray = _to_gray_numpy(image)   # (H, W)

    # Laplacian kernel (Definition 1.2 discrete form)
    lap_kernel = np.array([[0,  1, 0],
                            [1, -4, 1],
                            [0,  1, 0]], dtype=np.float32)

    # Manual 2-D convolution via torch for consistency
    h, w    = gray.shape
    t       = torch.from_numpy(gray).float().view(1, 1, h, w)
    k       = torch.from_numpy(lap_kernel).view(1, 1, 3, 3)
    lap_map = F.conv2d(t, k, padding=1).squeeze().numpy()   # (H, W)

    return float(lap_map.var())



# 2.  Tenengrad Criterion  (TEN) 


def tenengrad(image: torch.Tensor) -> float:
   
    gray = _to_gray_numpy(image)
    h, w = gray.shape
    t    = torch.from_numpy(gray).float().view(1, 1, h, w)

    # Sobel kernels (Definition 1.4)
    Sx = torch.tensor([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    Sy = torch.tensor([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)

    Gx = F.conv2d(t, Sx, padding=1).squeeze().numpy()
    Gy = F.conv2d(t, Sy, padding=1).squeeze().numpy()

    # Gradient magnitude M(x,y) = sqrt(Gx^2 + Gy^2) 
    M = np.sqrt(Gx ** 2 + Gy ** 2)

   
    return float(M.var())



# 3.  High-Frequency Energy Ratio  (HFE)  


def high_freq_energy_ratio(image: torch.Tensor, alpha: float = 0.1) -> float:
    
    gray = _to_gray_numpy(image)   # (H, W) in [0,1]
    H, W = gray.shape

   
    F_shift = np.fft.fftshift(np.fft.fft2(gray))
    mag     = np.abs(F_shift)      # |I_tilde(u,v)|

    # Low-frequency disc mask  
    r    = int(alpha * min(W, H))
    cy   = H // 2
    cx   = W // 2
    v_idx, u_idx = np.ogrid[:H, :W]
    dist_sq      = (v_idx - cy) ** 2 + (u_idx - cx) ** 2
    low_mask     = dist_sq <= r ** 2   # True inside disc
    high_mask    = ~low_mask           # True in annular region

    total_energy = mag.sum()
    if total_energy == 0:
        return 0.0

   
    return float(mag[high_mask].sum() / total_energy)


# 4.  Mean Local Standard Deviation  (MLSD)   


def mean_local_std(image: torch.Tensor, window_size: int = 7) -> float:
   
    gray = _to_gray_numpy(image)
    h, w = gray.shape
    t    = torch.from_numpy(gray).float().view(1, 1, h, w)

    # Box filter weights  
    w2    = window_size * window_size
    k     = torch.ones(1, 1, window_size, window_size) / w2
    pad   = window_size // 2

    # mu_w(x,y) = box filter of I
    mu    = F.conv2d(t, k, padding=pad)

    # mu_w2(x,y) = box filter of I^2
    mu2   = F.conv2d(t ** 2, k, padding=pad)

    # local variance = max(0, mu_w2 - mu_w^2)
    var   = torch.clamp(mu2 - mu ** 2, min=0.0)

    # local std
    sigma = var.sqrt()

    # F_MLSD = mean of sigma over all pixels  
    return float(sigma.mean().item())



# 5.  GLCM Contrast  (FCON) 


def glcm_contrast(
    image:    torch.Tensor,
    Q:        int   = 64,
    distance: int   = 1,
    angle_deg: float = 0.0,
) -> float:
  
    gray = _to_gray_numpy(image)   # (H, W) in [0,1]
    H, W = gray.shape

    # Quantise to Q grey levels  
    IQ = (gray * (Q - 1)).astype(np.int32).clip(0, Q - 1)

    # Displacement vector  
    angle_rad = np.deg2rad(angle_deg)
    dx = int(round(distance * np.cos(angle_rad)))
    dy = int(round(distance * np.sin(angle_rad)))

    # Build unnormalised GLCM C_ij
    C = np.zeros((Q, Q), dtype=np.float64)
    for y in range(H):
        for x in range(W):
            ny = y + dy
            nx = x + dx
            if 0 <= ny < H and 0 <= nx < W:
                i = IQ[y, x]
                j = IQ[ny, nx]
                C[i, j] += 1

    # Symmetrise: P_ij <- (C_ij + C_ji) / 2  then normalise
    C = (C + C.T) / 2.0
    total = C.sum()
    if total == 0:
        return 0.0
    P = C / total   # normalised GLCM

    # F_CON = sum_ij (i-j)^2 * P_ij  
    i_idx, j_idx = np.meshgrid(np.arange(Q), np.arange(Q), indexing='ij')
    contrast = float(np.sum((i_idx - j_idx) ** 2 * P))
    return contrast



#  compute all 5 metrics at once


def compute_all_metrics(image: torch.Tensor) -> dict[str, float]:
    
    return {
        "VoL":  variance_of_laplacian(image),
        "TEN":  tenengrad(image),
        "HFE":  high_freq_energy_ratio(image),
        "MLSD": mean_local_std(image),
        "FCON": glcm_contrast(image),
    }


def compute_metrics_batch(images: torch.Tensor) -> dict[str, list[float]]:
   
    results: dict[str, list[float]] = {
        "VoL": [], "TEN": [], "HFE": [], "MLSD": [], "FCON": []
    }
    for i in range(images.shape[0]):
        m = compute_all_metrics(images[i])
        for key in results:
            results[key].append(m[key])
    return results




