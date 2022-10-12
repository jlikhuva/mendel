from tkinter.messagebox import NO
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistFeedForwardClassifier(nn.Module):
    """
    A simple, feed-forward, fully-connected neural
    network for classifying MNIST digits
    """
    def __init__(self) -> None:
        super(MnistFeedForwardClassifier, self).__init__()
        self.flatten_image = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    
    def forward(self, image) -> torch.Tensor:
        x = self.flatten_image(image)
        logits = self.network(x)
        return logits