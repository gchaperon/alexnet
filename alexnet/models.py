import itertools
import torch
import torch.nn as nn
import typing as tp
import pytorch_lightning as pl


class _Repeat(nn.Module):
    repeat_args: tp.Sequence[int]

    def __init__(self, *repeat_args: int):
        super().__init__()
        self.repeat_args = repeat_args

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.repeat(*self.repeat_args)

    def extra_repr(self) -> str:
        return ", ".join(map(repr, self.repeat_args))


class AlexNet(nn.Module):
    INPUT_SIZE = 224

    def __init__(self, nclasses: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.nclasses = nclasses
        self.dropout = dropout
        # comments refer to the names in
        # https://github.com/akrizhevsky/cuda-convnet2/blob/master/layers/layers-imagenet-2gpu-model.cfg
        # I took the hparams from there because I found that was the net
        # closest to the one described in the paper

        # NOTE: model parallel is emulated by the `groups` option in `nn.Conv2d`.
        # therefore naming becomes (conv1a, conv1b) -> conv1
        # also, remember to repeat the input where necessary to emulate multi gpu

        groups = 2
        self.features = nn.Sequential(
            _Repeat(1, 2, 1, 1),
            nn.Conv2d(
                in_channels=3 * groups,
                out_channels=48 * groups,
                kernel_size=11,
                stride=4,
                padding=0,
                groups=groups,
            ),
            nn.LocalResponseNorm(size=5, alpha=5 * 10e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=48 * groups,
                out_channels=128 * groups,
                kernel_size=5,
                stride=1,
                padding=2,
                groups=groups,
            ),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=5 * 10e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(
                in_channels=128 * groups,
                out_channels=192 * groups,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=192 * groups,
                out_channels=192 * groups,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=groups,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=192 * groups,
                out_channels=128 * groups,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=groups,
            ),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, nclasses),
        )
        self.reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        features = self.features(input)
        flattened = torch.flatten(features, start_dim=1, end_dim=-1)
        logits: torch.Tensor = self.classifier(flattened)
        return logits

    def reset_parameters(self) -> None:
        for name, parameter in self.named_parameters():
            if "weight" in name:
                nn.init.normal_(parameter, mean=0.0, std=0.01)
            elif "bias" in name:
                nn.init.constant_(parameter, 0.0)

        # special cases where bias is initialized with 1
        for module in itertools.chain(
            (self.features[i] for i in (5, 11, 13)),
            (m for m in self.classifier if isinstance(m, nn.Linear)),
        ):
            nn.init.constant_(module.bias, 1.0)


class _OptimizerOpts(tp.TypedDict):
    lr: float
    momentum: float
    weight_decay: float


class LitAlexNet(AlexNet, pl.LightningModule):
    optimizer_opts: _OptimizerOpts

    def __init__(
        self, nclasses: int, optimizer_opts: _OptimizerOpts, dropout: float = 0.0
    ) -> None:
        super().__init__(nclasses, dropout)
        self.save_hyperparameters()

        self.loss = nn.CrossEntropyLoss()
        self.optimizer_opts = optimizer_opts

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        input, target = batch
        logits = self(input)
        loss: torch.Tensor = self.loss(logits, target)
        self.log("train/loss", loss)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        input, target = batch
        logits = self(input)
        loss = self.loss(logits, target)
        self.log("val/loss", loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), **self.optimizer_opts)
