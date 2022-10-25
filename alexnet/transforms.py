import numpy as np
import typing as tp
import torch
from torchvision.transforms import (
    Compose,
    CenterCrop,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    TenCrop,
    ToTensor,
    ToPILImage,
    Normalize,
)

__all__ = [
    "Compose",
    "CenterCrop",
    "RandomCrop",
    "RandomHorizontalFlip",
    "Resize",
    "TenCrop",
    "ToTensor",
    "ToPILImage",
    "Normalize",
    "PCAAugment",
]


class PCAAugment:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor shape: C x H x W
        assert (tensor >= 0).all() and (tensor <= 1).all()
        assert isinstance(tensor, torch.Tensor), "PCAAugment applies only to tensors"
        nchannels = tensor.shape[0]
        pixels = tensor.view(nchannels, -1)
        # substracting mean is the first step to PCA
        pixels = pixels - torch.mean(pixels, dim=1, keepdim=True)
        # shape: C x C
        corr = torch.corrcoef(pixels)
        # C          C x C
        eigenvalues, eigenvectors = map(torch.from_numpy, np.linalg.eig(corr))
        assert torch.isreal(eigenvalues).all() and torch.isreal(eigenvectors.all())
        # C
        alpha = 0.1 * torch.randn(3)
        # C
        delta: torch.Tensor = eigenvectors @ (alpha * eigenvalues)
        return torch.clamp(tensor + delta[:, None, None], 0, 1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Stack:
    dim: int

    def __init__(self, dim: int = 0) -> None:
        self.dim = dim

    def __call__(self, tensors: tp.List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(tensors, dim=self.dim)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim})"
