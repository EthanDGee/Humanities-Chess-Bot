import json
from typing import List

import torch
from torch.optim import Adam
from torch.utils.data import Dataloader

from dataloader import ChessDataSet


class Trainer:
    def __init__(self, parameter_path: str, model):
        with open(parameter_path) as file:
            values = json.load(file)

            hyperparameters = values["hyperparameters"]
            database_info = values["database_info"]

        # Model
        self.model = model

        # stable parameters
        self.num_epochs: int = hyperparameters["num_epochs"]
        self.batch_size: int = hyperparameters["batch_size"]
        self.num_workers: int = hyperparameters["num_workers"]

        # searchable parameters
        self.learning_rates: List = hyperparameters["learning_rates"]
        self.decay_rates: List = hyperparameters["decay_rates"]
        self.betas: List = hyperparameters["betas"]
        self.momementums: List = hyperparameters["momementums"]

        # current parameters
        self.current_lr: float = self.learning_rates[0]
        self.current_decay_rate: float = self.decay_rates[0]
        self.current_beta: float = self.betas[0]
        self.current_momentum: float = self.momementums[0]

        # data loader
        self.train_dataloader = ChessDataSet()
        self.val_dataloader = ChessDataSet()

    def train(self):
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(
            self.model.parameters(),
            lr=self.current_lr,
            weight_decay=self.current_decay_rate,
            betas=(self.current_beta, self.current_momentum),
        )

        # Training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0

            for batch_x, batch_y in self.train_dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_x, batch_y in self.val_dataloader:
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

            avg_train_loss = train_loss / len(self.train_dataloader)
            avg_val_loss = val_loss / len(self.val_dataloader)
            val_accuracy = 100 * correct / total

            print(
                f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
            )
