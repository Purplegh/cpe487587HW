

import os
import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class FocalTverskyLoss(nn.Module):
    
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self.smooth = smooth

    def forward(self, preds, targets):
        probs   = torch.sigmoid(preds).view(-1)
        targets = targets.view(-1)

        TP = (probs * targets).sum()
        FP = ((1 - targets) * probs).sum()
        FN = (targets * (1 - probs)).sum()

        tversky_index = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )
        return (1.0 - tversky_index) ** self.gamma




class SpeedFeatureBlock(nn.Module):
    

    def __init__(self, in_features=44, out_features=256, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (N, in_features)
        return self.proj(x)   # (N, out_features)




class ACCNet(nn.Module):


    def __init__(self, window_size=44, dropout=0.3):
        super().__init__()
        self.window_size = window_size

        # custom layer (satisfies assignment requirement)
        self.feature_block = SpeedFeatureBlock(
            in_features  = window_size,
            out_features = 256,
            dropout      = dropout,
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (N, window_size)
        x = self.feature_block(x)   # (N, 128)
        x = self.fc(x)              # (N, 1)
        return x.squeeze(1)         # (N,)




class ACCDataset(Dataset):
    

    def __init__(self, csv_pairs, k=10, norm_mean=None, norm_std=None):
        self.k         = k
        self.norm_mean = norm_mean
        self.norm_std  = norm_std

        all_features = []
        all_labels   = []

        for speed_path, acc_path in csv_pairs:
            feats, labels = self._load_one_experiment(speed_path, acc_path)
            if feats is not None:
                all_features.append(feats)
                all_labels.append(labels)

        if len(all_features) == 0:
            raise RuntimeError("No valid CSV pairs were loaded.")

        self.X = np.concatenate(all_features, axis=0).astype(np.float32)
        self.y = np.concatenate(all_labels,   axis=0).astype(np.float32)

    def _load_one_experiment(self, speed_path, acc_path):
        try:
            speed_df = pd.read_csv(speed_path)
            acc_df   = pd.read_csv(acc_path)
        except Exception as e:
            print(f"[ACCDataset] Could not read files: {e}")
            return None, None

        # speed: keep Time and Message only, convert km/h -> m/s
        speed_df = speed_df[['Time', 'Message']].copy()
        speed_df.rename(columns={'Message': 'speed_kmh'}, inplace=True)
        speed_df['speed_ms'] = speed_df['speed_kmh'] / 3.6
        speed_df.sort_values('Time', inplace=True)
        speed_df.reset_index(drop=True, inplace=True)

        # acc: keep Time and Message only, binarize label
        acc_df = acc_df[['Time', 'Message']].copy()
        acc_df.rename(columns={'Message': 'acc_status'}, inplace=True)
        acc_df.sort_values('Time', inplace=True)
        acc_df.reset_index(drop=True, inplace=True)
        acc_df['label'] = (acc_df['acc_status'] == 6).astype(int)

        # ZOH: for each speed timestamp find most recent acc timestamp <= it
        merged = pd.merge_asof(
            speed_df,
            acc_df[['Time', 'label']],
            on='Time',
            direction='backward'
        )
        merged.dropna(subset=['label'], inplace=True)
        merged.reset_index(drop=True, inplace=True)

        speeds = merged['speed_ms'].values
        labels = merged['label'].values.astype(np.float32)

        # normalise speed
        if self.norm_mean is not None and self.norm_std is not None:
            speeds = (speeds - self.norm_mean) / (self.norm_std + 1e-8)

        # derive additional feature groups from speed signal
        # acceleration: 1st finite difference of speed
        accel = np.diff(speeds, prepend=speeds[0]).astype(np.float32)
        # jerk: 2nd finite difference of speed
        jerk  = np.diff(accel,  prepend=accel[0]).astype(np.float32)
        # rolling std over k samples (local speed variability)
        rolling_std = np.array([
            speeds[max(0, i - self.k): i + 1].std()
            for i in range(len(speeds))
        ], dtype=np.float32)

        # sliding window: 4 groups x (k+1) features
        n = len(speeds)
        if n <= self.k:
            print(f"[ACCDataset] Experiment too short ({n} samples), skipping.")
            return None, None

        X_rows = []
        y_rows = []
        for t in range(self.k, n):
            s  = speeds[t - self.k: t + 1][::-1].copy()       # (k+1,) speed
            a  = accel[t - self.k: t + 1][::-1].copy()        # (k+1,) accel
            j  = jerk[t - self.k: t + 1][::-1].copy()         # (k+1,) jerk
            rs = rolling_std[t - self.k: t + 1][::-1].copy()  # (k+1,) std
            window = np.concatenate([s, a, j, rs])             # 4*(k+1) = 44
            X_rows.append(window)
            y_rows.append(labels[t])

        return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])



def build_csv_pairs(data_dir):
  
    speed_files = sorted(glob.glob(
        os.path.join(data_dir, '*decoded_wheel_speed_fl.csv')
    ))

    pairs = []
    for sf in speed_files:
        basename  = os.path.basename(sf)
        timestamp = basename.replace('_decoded_wheel_speed_fl.csv', '')
        acc_file  = os.path.join(data_dir, f'{timestamp}_decoded_acc_status.csv')
        if os.path.exists(acc_file):
            pairs.append((sf, acc_file))
        else:
            print(f"[build_csv_pairs] No ACC file for: {sf} - skipping.")

    print(f"[build_csv_pairs] Found {len(pairs)} valid experiment pairs.")
    return pairs



class ACCTrainer:
  

    def __init__(self, model, train_loader, val_loader, device,
                 lr=3e-4, weight_decay=1e-4,
                 alpha=0.3, beta=0.7, gamma=0.75):

        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device

        self.criterion = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # reduces LR by factor 0.5 if val_acc does not improve for 10 epochs
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode     = 'max',
            factor   = 0.5,
            patience = 10,
            min_lr   = 1e-6,
        )

        self.history = {
            'train_loss': [], 'train_acc': [], 'val_acc': []
        }

    def _accuracy(self, logits, labels):
        
        preds = (torch.sigmoid(logits) >= 0.5).float()
        tp    = ((preds == 1) & (labels == 1)).float().sum()
        tn    = ((preds == 0) & (labels == 0)).float().sum()
        pos   = labels.sum().clamp(min=1)
        neg   = (1 - labels).sum().clamp(min=1)
        return (tp / pos + tn / neg).item() / 2

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_acc  = 0.0

        for batch_idx, (X, y) in enumerate(self.train_loader):
            X = X.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(X)
            loss   = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_acc  += self._accuracy(logits, y)

            # print every 10th batch as required by assignment
            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx + 1}"
                      f"/{len(self.train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(self.train_loader)
        avg_acc  = total_acc  / len(self.train_loader)
        return avg_loss, avg_acc

    def validate(self):
        self.model.eval()
        total_acc = 0.0
        with torch.no_grad():
            for X, y in self.val_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                logits = self.model(X)
                total_acc += self._accuracy(logits, y)
        return total_acc / len(self.val_loader)

    def fit(self, num_epochs):
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_acc = self.validate()

            self.scheduler.step(val_acc)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            print(f"[Epoch {epoch:4d}/{num_epochs}]  "
                  f"Loss: {train_loss:.4f}  "
                  f"Train Acc: {train_acc * 100:.2f}%  "
                  f"Val Acc: {val_acc * 100:.2f}%  "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

    def save_onnx(self, save_path, window_size=44):
        self.model.eval()
        dummy_input = torch.zeros(1, window_size).to(self.device)
        torch.onnx.export(
            self.model,
            dummy_input,
            save_path,
            input_names   = ['speed_window'],
            output_names  = ['logit'],
            dynamic_axes  = {
                'speed_window': {0: 'batch_size'},
                'logit':        {0: 'batch_size'},
            },
            opset_version = 14,
        )
        print(f"[ACCTrainer] ONNX model saved -> {save_path}")