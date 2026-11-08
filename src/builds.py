import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# constructing a configurable neural network
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 20,
        width: int = 64,
        depth: int = 8,
        activation: str = "relu",
        init: str = "xavier",
    ):
        super().__init__()
        layers = []
        in_features = input_dim

        for _ in range(depth):
            layers.append(nn.Linear(in_features, width))
            layers.append(make_activation(activation))
            in_features = width

        layers.append(nn.Linear(in_features, 1))  # logit output
        self.network = nn.Sequential(*layers)
        initialize_linear_layers(self, init, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# function to control randomness
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# function to select behavior dynamically
def make_activation(name: str) -> nn.Module:
    activations = {
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]()

# function to controll initial weights
def initialize_linear_layers(model: nn.Module, scheme: str, activation: str) -> None:
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue

        if scheme == "xavier":
            nn.init.xavier_uniform_(module.weight)
        elif scheme == "he":
            # Kaiming/He initialization is primarily intended for ReLU-like activations.
            nonlinearity = "relu" if activation == "relu" else "leaky_relu"
            nn.init.kaiming_uniform_(module.weight, nonlinearity=nonlinearity)
        elif scheme == "default":
            # Keep PyTorch's default Linear initialization for a baseline.
            pass
        else:
            raise ValueError(f"Unknown initialization: {scheme}")

        if module.bias is not None and scheme != "default":
            nn.init.zeros_(module.bias)

# the core of the research experiment
def measure_gradients(
    depth: int,
    activation: str,
    init: str,
    seed: int,
    batch_size: int = 128,
    input_dim: int = 20,
    width: int = 64,
) -> list[dict]:
    set_seed(seed)
    x = torch.randn(batch_size, input_dim, device=DEVICE)
    # Fixed random binary targets make a reproducible diagnostic objective.
    y = torch.randint(0, 2, (batch_size, 1), dtype=torch.float32, device=DEVICE)

    model = MLP(input_dim, width, depth, activation, init).to(DEVICE)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss = nn.BCEWithLogitsLoss()(logits, y)
    loss.backward()

    rows = []
    linear_index = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            grad = module.weight.grad
            if grad is None:
                grad_norm = float("nan")
                grad_rms = float("nan")
            else:
                grad_norm = grad.detach().norm(2).item()
                grad_rms = grad.detach().pow(2).mean().sqrt().item()
            rows.append({
                "seed": seed,
                "depth": depth,
                "activation": activation,
                "init": init,
                "layer_index": linear_index,
                "is_output_layer": linear_index == depth,
                "weight_grad_l2": grad_norm,
                "weight_grad_rms": grad_rms,
                "loss": loss.item(),
            })
            linear_index += 1
    return rows