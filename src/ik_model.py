import torch
import torch.nn as nn

class IKNet(nn.Module):
    """Simple feedforward neural network for inverse kinematics. 
    
    Heuristically estimating hidden layer size based on degrees of freedom
    Simplifying the problem with an input of 1,0,0, the output would require the generation new coords (either in linear or angular space) for each joint. 
    The thought process for that layer is anticipating a mapping from 3D coords to angular space so Tanh was chosen for the activation function.
    Then it would need to convert those coords into angles for each joint which occurs at a mix of the second hidden layer where it utilizes relu to introduce non-linearity.
    """
    def __init__(self, input_dim=3, output_dim=7):
        hidden_dim = output_dim * 3 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)