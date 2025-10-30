from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # this is identical to the stock fish - engineered features and class based output layer
        self.layers = nn.Sequential(
            nn.Linear(67, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 4164),
            nn.Softmax(),
        )

    def forward(self, x):
        return self.layers(x)


model = NeuralNetwork()
