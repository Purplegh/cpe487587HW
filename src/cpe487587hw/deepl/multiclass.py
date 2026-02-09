
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
