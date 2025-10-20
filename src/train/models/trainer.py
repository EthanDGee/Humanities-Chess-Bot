import json
from typing import List

from torch.optim import Adam
from torch.utils.data import Dataloader

from data_loader import Data


class Trainer:
    def __init__(self, parameter_path: str):
        with open(parameter_path) as file:
            values = json.load(file)

            hyperparameters = values["hyperparameters"]
            database_info = values["database_info"]

        # stable parameters
        self.num_epochs: int = hyperparameters["num_epochs"]
        self.batch_size: int = hyperparameters["batch_size"]
        self.num_workers: int = hyperparameters["num_workers"]

        # searchable parameters
        self.learning_rates: List = hyperparameters["learning_rates"]
        self.decay_rates: List = hyperparameters["decay_rates"]
        self.betas: List = hyperparameters["betas"]
        self.momementums: List = hyperparameters["momementums"]

        # data loader
        self.dataloader = Data()

    def train():
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(
            model.parameters(), lr=lr, weight_decay=decay_rate, betas=(beta, momentum)
        )

        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total

            print(
                f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
            )
