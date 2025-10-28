import json
import random
import time

import torch
from dataloader import ChessDataSet
from torch import nn
from torch.optim import Adam


class Trainer:
    def __init__(self, parameter_path: str, model):
        with open(parameter_path) as file:
            values = json.load(file)

            hyperparameters = values["hyperparameters"]
            database_info = values["database_info"]
            checkpoints = values["checkpoints"]

        # Model
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        # stable parameters
        self.num_epochs: int = hyperparameters["num_epochs"]
        self.batch_size: int = hyperparameters["batch_size"]
        self.num_workers: int = hyperparameters["num_workers"]

        # searchable parameters
        self.learning_rates: list = hyperparameters["learning_rates"]
        self.decay_rates: list = hyperparameters["decay_rates"]
        self.betas: list = hyperparameters["betas"]
        self.momementums: list = hyperparameters["momementums"]

        # current parameters
        self.current_lr: float = self.learning_rates[0]
        self.current_decay_rate: float = self.decay_rates[0]
        self.current_beta: float = self.betas[0]
        self.current_momentum: float = self.momementums[0]

        # data loader
        total_instances = database_info["num_indexes"]
        self.train_dataloader = ChessDataSet(database_info["training_path"], total_instances)
        self.val_dataloader = ChessDataSet(database_info["validation_path"], total_instances * 0.2)

        # Model Checkpoints Path
        self.save_directory = checkpoints["directory"]
        self.auto_save_path = self.save_directory + "/check_points/"
        self.auto_save_interval = checkpoints["auto_save_interval"]
        self.final_save = self.save_directory + "/trained_models/"
        self.model_name = ""

    def _update_model_name(self):
        """
        Updates the model name to be a description of the current learning rate, decay rate,
        beta, and momentum values.

        Returns:
            None
        """
        updated_name = f"del_lr{self.current_lr}"
        updated_name += f"_decay{self.current_decay_rate}"
        updated_name += f"_beta{self.current_beta}"
        updated_name += f"_momentum{self.current_momentum}"

        self.model_name = updated_name

    def train(self):
        """
        Trains the model using the current hyper parameters saving the model periodically
        to a check point dirctory based on self.auto_save_interval as well as training
        information before saving the final model

        Returns:
            None
        """
        # Define loss function and optimizer
        optimizer = Adam(
            self.model.parameters(),
            lr=self.current_lr,
            weight_decay=self.current_decay_rate,
            betas=(self.current_beta, self.current_momentum),
        )

        last_save_time = time.time()
        # Training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0

            for batch_x, batch_y in self.train_dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # check for auto save
                if time.time() - last_save_time >= self.auto_save_interval:
                    self._save_model()
                    last_save_time = time.time()

            avg_train_loss = train_loss / len(self.train_dataloader)
            avg_val_loss, val_accuracy = self._validation_loss()
            print(
                f"Epoch: {epoch}/{self.num_epochs} | Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
            )

    def _validation_loss(self):
        """
        Computes the validation loss and accuracy for the model.

        Returns:
            tuple: A tuple containing two elements:
                - avg_val_loss (float): the average validation loss
                - val_accuracy (float): the percentage of correct predictions
        """
        # Validation
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        avg_val_loss = val_loss / len(self.val_dataloader)
        val_accuracy = 100 * correct / total

        with torch.no_grad():
            for batch_x, batch_y in self.val_dataloader:
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        return avg_val_loss, val_accuracy

    def randomize_hyperparameters(self):
        """
        Randomizes and assigns hyperparameter values from the lists of possible values.

        This method selects random values for learning rate, decay rate, beta,
        and momentum from their respective lists and assigns them to the object's
        instance variables.

        Returns:
            None
        """
        self.current_lr = random.choice(self.learning_rates)
        self.current_decay_rate = random.choice(self.decay_rates)
        self.current_beta = random.choice(self.betas)
        self.current_momentum = random.choice(self.momementums)

    def random_search(self, iterations: int):
        """
        Conducts a random search for hyperparameter optimization by iteratively testing
        random configurations, evaluating their performance, and updating the best set
        of hyperparameters based on validation loss.

        Args:
            iterations (int): The number of random configurations to search over

        Returns:
            None
        """
        best_val_loss = float("inf")
        best_hyperparameters = {}

        for _ in range(iterations):
            self.randomize_hyperparameters()
            print(
                f"Testing with LR: {self.current_lr}, Decay: {self.current_decay_rate}, Beta: {self.current_beta}, Momentum: {self.current_momentum}"
            )
            self.train()
            self._save_model()
            avg_val_loss, _ = self._validation_loss()
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_hyperparameters = {
                    "learning_rate": self.current_lr,
                    "decay_rate": self.current_decay_rate,
                    "beta": self.current_beta,
                    "momentum": self.current_momentum,
                }
                print(
                    f"New best hyperparameters found: {best_hyperparameters} with Val Loss: {best_val_loss:.4f}"
                )

        print(f"Best Hyperparameters: {best_hyperparameters} with Val Loss: {best_val_loss:.4f}")

    def _save_model(self, auto_save: bool = True):
        """
        Saves the model state to a file in the appropriate directory.

        This method saves the state of the model to either the check_points
        directory for the model or the final output directoryd depending on
        whether the auto-save option is enabled or not.

        Args:
            auto_save (bool): Indicates whether to save the model in the auto-save
                directory or the final save directory. Defaults to True.

        Returns:
            train_dataloader
        """
        if auto_save:
            save_directory = self.auto_save_path + self.model_name
        else:
            save_directory = self.final_save + self.model_name
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        torch.save(self.model.state_dict(), f"{save_directory}/{timestamp}.pth")
