import numpy as np
import typing as tp
import PIL.Image
import torch
from torchvision.transforms import (
    Compose,
    CenterCrop,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    FiveCrop,
    TenCrop,
    ToTensor,
    ToPILImage,
    Normalize,
)

__all__ = [
    "Compose",
    "CenterCrop",
    "Lambda",
    "RandomCrop",
    "RandomHorizontalFlip",
    "Resize",
    "FiveCrop",
    "TenCrop",
    "ToTensor",
    "ToPILImage",
    "Normalize",
    "PCAAugment",
    "Unsqueeze",
    "ToRGB",
]


class PCAAugment:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor shape: C x H x W
        assert (tensor >= 0).all() and (tensor <= 1).all()
        assert isinstance(tensor, torch.Tensor), "PCAAugment applies only to tensors"
        array = tensor.numpy()
        nchannels = array.shape[0]
        pixels = array.reshape(nchannels, -1)
        # substracting mean is the first step to PCA
        pixels = pixels - np.mean(pixels, axis=1, keepdims=True)
        # shape: C x C
        corr = np.corrcoef(pixels)
        try:
            # C          C x C
            eigenvalues, eigenvectors = np.linalg.eig(corr)
            assert np.isrealobj(eigenvalues) and np.isrealobj(eigenvectors)
            # C
            alpha = 0.1 * np.random.randn(nchannels)
            # C
            delta = eigenvectors @ (alpha * eigenvalues)
        except np.linalg.LinAlgError:
            # C
            delta = np.zeros(nchannels)

        return torch.from_numpy(
            np.clip(array + delta[:, None, None], 0.0, 1.0).astype(np.float32)
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ToRGB:
    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        return img.convert("RGB")

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


class Unsqueeze:
    dim: int

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.unsqueeze(tensor, dim=self.dim)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim})"
