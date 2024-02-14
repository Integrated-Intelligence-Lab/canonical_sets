from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from canonical_sets.data import DataSet
from canonical_sets.models import ClassifierPT


class AdultClassifier:
    def __init__(self, data, seed=42):
        self.data = data
        self.seed = torch.manual_seed(seed)

        self.train_dataset = DataSet(data.train_data, data.train_labels)
        self.test_dataset = DataSet(data.test_data, data.test_labels)

        self.train_dl = DataLoader(
            self.train_dataset, batch_size=32, shuffle=True
        )
        self.test_dl = DataLoader(
            self.test_dataset, batch_size=32, shuffle=False
        )

        self.input_dim = len(data.train_data.columns)

        self.model = ClassifierPT(self.input_dim, 2)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())

        self.epochs = 0

    def train_adult_classifier(self, epochs=1):
        curves = []

        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer

        for epoch in range(epochs):
            model.train()

            train_loss = 0
            num_samples = 0

            for x, y in self.train_dl:
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.detach().cpu()
                num_samples += outputs.size(0)

            train_loss /= len(self.train_dataset)

            test_loss = 0
            num_correct = 0
            num_samples = 0

            model.eval()

            with torch.inference_mode():
                for x, y in self.test_dl:
                    scores = model(x)
                    test_loss += criterion(scores, y)
                    predictions = scores.argmax(1)
                    num_correct += (predictions == y.argmax(1)).sum()
                    num_samples += predictions.size(0)

                test_acc = float(num_correct) / float(num_samples)
                test_loss /= num_samples

                print(
                    f"Epoch: {epoch} | Train_loss: {train_loss}"
                    f"| Test_loss: {test_loss} | Test_accuracy: {test_acc}"
                )

                curves.append(
                    {
                        "Epoch": epoch,
                        "Train_loss": train_loss,
                        "Test_loss": test_loss,
                        "Test_acc": test_acc,
                    }
                )

        self.curves = curves
        self.epochs += epochs

        return curves

    def predict(self, data: pd.DataFrame):
        data = data.copy()

        # Convert boolean columns to floats
        for column in data.columns:
            if data[column].dtype == bool:
                data[column] = data[column].astype("uint")

        # Convert DataFrame to NumPy array
        x = data.to_numpy()

        # Convert NumPy array to PyTorch tensor
        x = torch.tensor(x).to(dtype=torch.float32)

        self.model.eval()
        with torch.inference_mode():
            logits = self.model(x)

        return logits

    def save_model(self, path: str, model_name="adultclassifier.pth"):
        # Create target directory
        target_dir_path = Path(path)
        target_dir_path.mkdir(parents=True, exist_ok=True)

        # Create model save path
        assert model_name.endswith(".pth") or model_name.endswith(
            ".pt"
        ), "model_name should end with '.pt' or '.pth'"
        model_save_path = target_dir_path / model_name

        # Save the model state_dict()
        print(f"[INFO] Saving model to: {model_save_path}")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "curves": self.curves,
                "epochs": self.epochs,
            },
            model_save_path,
        )

    def load_model(self, path):
        assert path.endswith(".pth") or path.endswith(
            ".pt"
        ), "path to model should end with '.pt' or '.pth'"

        # Load the checkpoint
        checkpoint = torch.load(path)

        # Load states into the model and optimizer
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.curves = checkpoint["curves"]
        self.epochs = checkpoint["epochs"]
