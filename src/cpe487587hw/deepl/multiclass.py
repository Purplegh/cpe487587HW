
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

class SimpleNN(nn.Module):
    def __init__(self, in_features, num_classes):
        super(SimpleNN, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        self.fc1 = nn.Linear(self.in_features, 3)
        self.fc2 = nn.Linear(3, 4)
        self.fc3 = nn.Linear(4, 5)
        self.fc4 = nn.Linear(5, self.num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x



class ClassTrainer:
    def __init__(
        self,
        X_train,
        Y_train,
        model,
        eta=0.001,
        epochs=100,
        loss_fn=None,
        optimizer_cls=optim.Adam,
        device=None
    ):
        self.device = device if device else (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.X_train = X_train.to(self.device)
        self.Y_train = Y_train.to(self.device)
        self.model = model.to(self.device)

        self.eta = eta
        self.epoch = epochs

        self.loss = loss_fn if loss_fn else nn.CrossEntropyLoss()
        self.optimizer = optimizer_cls(self.model.parameters(), lr=self.eta)

        self.loss_vector = torch.zeros(self.epoch)
        self.accuracy_vector = torch.zeros(self.epoch)

    # ---------------- TRAIN ----------------
    def train(self):
        self.model.train()

        for ep in range(self.epoch):
            self.optimizer.zero_grad()

            out = self.model(self.X_train)
            loss_val = self.loss(out, self.Y_train)

            loss_val.backward()
            self.optimizer.step()

            preds = torch.argmax(out, dim=1)
            acc = (preds == self.Y_train).float().mean()

            self.loss_vector[ep] = loss_val.item()
            self.accuracy_vector[ep] = acc.item()

        return self.loss_vector, self.accuracy_vector

    # ---------------- TEST ----------------
    def test(self, X_test, Y_test):
        self.model.eval()

        with torch.no_grad():
            X_test = X_test.to(self.device)
            out = self.model(X_test)
            preds = torch.argmax(out, dim=1)

        return preds.cpu(), Y_test.cpu()

    # ---------------- PREDICT ----------------
    def predict(self, X):
        self.model.eval()

        with torch.no_grad():
            X = X.to(self.device)
            out = self.model(X)
            preds = torch.argmax(out, dim=1)

        return preds.cpu()

    # ---------------- SAVE ----------------
    def save(self, filename="multiclass_model.onnx"):
        sample_input = self.X_train[:1]
        torch.onnx.export(self.model, sample_input, filename, opset_version=11)


    # ---------------- EVALUATION ----------------
    def evaluation(self, X_test, Y_test):
        # Train metrics
        train_preds = torch.argmax(self.model(self.X_train), dim=1).cpu()
        y_train = self.Y_train.cpu()

        # Test metrics
        test_preds, y_test = self.test(X_test, Y_test)

        metrics = {
            "train_accuracy": accuracy_score(y_train, train_preds),
            "train_precision": precision_score(y_train, train_preds, average="macro"),
            "train_recall": recall_score(y_train, train_preds, average="macro"),
            "train_f1": f1_score(y_train, train_preds, average="macro"),
            "test_accuracy": accuracy_score(y_test, test_preds),
            "test_precision": precision_score(y_test, test_preds, average="macro"),
            "test_recall": recall_score(y_test, test_preds, average="macro"),
            "test_f1": f1_score(y_test, test_preds, average="macro"),
        }

        # Loss plot
        plt.figure()
        plt.plot(self.loss_vector)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

        # Accuracy plot
        plt.figure()
        plt.plot(self.accuracy_vector)
        plt.title("Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.show()

        # Confusion matrix
        cm = confusion_matrix(y_test, test_preds)
        ConfusionMatrixDisplay(cm).plot()
        plt.title("Confusion Matrix (Test)")
        plt.show()

        return metrics
##___________________________________HW_03___________________________________________________________________________________

import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ConvLayer(nn.Module):
   

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ImageNetCNN(nn.Module):
    

    def __init__(self, num_classes: int):
        super().__init__()

        # Five convolutional blocks (Figure 1)
        self.conv_blocks = nn.Sequential(
            ConvLayer(3,   64),   # Block 1
            ConvLayer(64,  128),  # Block 2
            ConvLayer(128, 256),  # Block 3
            ConvLayer(256, 512),  # Block 4
            ConvLayer(512, 512),  # Block 5
        )

        # Global average pooling  (224 / 2^5 = 7  -> 1x1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully-connected head (FC Layer 1: ReLU + Dropout, FC Layer 2: logits)
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)          # (B, 512, H', W')
        x = self.global_avg_pool(x)      # (B, 512, 1, 1)
        x = torch.flatten(x, 1)          # (B, 512)
        x = self.classifier(x)           # (B, num_classes)
        return x

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNNTrainer:
    

    def __init__(
        self,
        model: ImageNetCNN,
        device: torch.device,
        criterion=None,
        optimizer=None,
        scheduler=None,
        output_dir: str = "outputs",
    ):
        self.model = model.to(device)
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

       
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

    
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=0.01,
                momentum=0.9,
                weight_decay=1e-4,
            )

       
        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )

      
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

   
    def _train_one_epoch(self, train_loader, val_loader, epoch: int):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            inputs = batch["pixel_values"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Print every 10th batch with epoch, loss, train acc, and val acc
            if (batch_idx + 1) % 10 == 0:
                train_acc = 100.0 * correct / total
                val_acc = self._validate(val_loader)
                self.model.train()  # switch back to train mode after validation
                print(
                    f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} "
                    f"| Loss: {running_loss/total:.4f} "
                    f"| Train Acc: {train_acc:.2f}% "
                    f"| Val Acc: {val_acc:.2f}%"
                )

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def _validate(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0

        for batch in val_loader:
            inputs = batch["pixel_values"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            outputs = self.model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return 100.0 * correct / total

    def train(self, train_loader, val_loader, num_epochs: int = 10):
       
        for epoch in range(num_epochs):
            t0 = time.time()
            train_loss, train_acc = self._train_one_epoch(train_loader, val_loader, epoch)
            val_acc = self._validate(val_loader)
            self.model.train()
            self.scheduler.step()

            elapsed = time.time() - t0
            print(
                f"\n[Epoch {epoch+1}/{num_epochs} complete] "
                f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
                f"| Val Acc: {val_acc:.2f}% | Time: {elapsed:.1f}s\n"
            )

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

        print("Training complete.")

    def save_onnx(self, onnx_path: str = None, input_shape=(1, 3, 224, 224)):
        
        if onnx_path is None:
            onnx_path = os.path.join(self.output_dir, "imagenet_cnn.onnx")

        self.model.eval()
        dummy_input = torch.randn(*input_shape).to(self.device)
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        print(f"ONNX model saved to: {onnx_path}")
        return onnx_path

    def save_checkpoint(self, path: str = None):
        """Save PyTorch model weights."""
        if path is None:
            path = os.path.join(self.output_dir, "imagenet_cnn.pth")
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved to: {path}")

    def plot_history(self, save_path: str = None):
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, "training_history.png")

        epochs = range(1, len(self.history["train_loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Loss subplot
        ax1.plot(epochs, self.history["train_loss"], "b-o", label="Train Loss")
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        # Accuracy subplot
        ax2.plot(epochs, self.history["train_acc"], "b-o", label="Train Acc")
        ax2.plot(epochs, self.history["val_acc"],   "r-s", label="Val Acc")
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Training history plot saved to: {save_path}")
        return save_path