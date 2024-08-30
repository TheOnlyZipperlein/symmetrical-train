import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the Generator
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.main(x)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.main(x)


class GANParameterOptimizer:
    def __init__(self, input_size:int):
        self.input_size = 100 # Size of the random noise vector
        self.hidden_size = 256 
        self.parameter_count = 3
        self.learning_rate = 0.0002
        self.reset_progress()

    def reset_progress(self):
        self.G = Generator(self.input_size, self.hidden_size, self.parameter_count)
        self.D = Discriminator(self.parameter_count, self.hidden_size, 1)

        self.criterion = nn.BCELoss()
        self.optimizerD = optim.Adam(self.D.parameters(), lr=self.learning_rate)
        self.optimizerG = optim.Adam(self.G.parameters(), lr=self.learning_rate)

    def train(self, X, y, num_epochs, wrapper):
        X = torch.Tensor(X)
        y = torch.Tensor(y).unsqueeze(-1)

        for epoch in range(num_epochs):
            # Train Discriminator
            d_pred = self.D(X)
            d_loss = self.criterion(d_pred, y)
            self.D.zero_grad()
            d_loss.backward()
            self.optimizerD.step()

            # Train Generator
            noise = torch.randn(len(X), self.input_size)
            X_gen = self.G(noise)
            y_gen = self.D(X_gen)

            y_tested = wrapper.evaluate(X_gen)
            y_tested = torch.tensor(np.array(y_tested).astype(dtype=np.float32)).unsqueeze(-1)
            g_loss = self.criterion(y_gen, y_tested)  # The generator tries to trick the discriminator

            self.G.zero_grad()
            g_loss.backward()
            self.optimizerG.step()

            wrapper.drop_models()
    def generate(sample_size):
        pass

        
