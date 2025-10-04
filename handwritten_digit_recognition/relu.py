import torch

def my_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.zeros_like(x), x)
