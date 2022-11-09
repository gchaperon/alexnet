import torch.nn.functional as F
import torchmetrics
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

        # NOTE: model parallel training is emulated by the `groups` option in
        # `nn.Conv2d`.
        # Some dimensions and layer arguments are taken from
        # https://github.com/akrizhevsky/cuda-convnet2/blob/master/layers/layers-imagenet-2gpu-model.cfg
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
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=5 * 10e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
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
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
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


class LitAlexNet(AlexNet, pl.LightningModule):
    optimizer_opts: _OptimizerOpts

    def __init__(
        self,
        nclasses: int,
        optimizer_opts: _OptimizerOpts,
        dropout: float = 0.0,
        extra_logging: bool = False,
    ) -> None:
        super().__init__(nclasses, dropout)
        self.save_hyperparameters()

        self.optimizer_opts = optimizer_opts
        self.extra_logging = extra_logging
        # for graph logging in tensorboard
        self.example_input_array = torch.randn(16, 3, 224, 224)

        metrics = torchmetrics.MetricCollection(
            {
                "error@1": 1 - torchmetrics.Accuracy(top_k=1),  # type:ignore[operator]
                "error@5": 1 - torchmetrics.Accuracy(top_k=5),  # type:ignore[operator]
            },
        )
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        input, target = batch
        logits = self(input)
        loss: torch.Tensor = F.cross_entropy(logits, target)
        self.log("train/loss", loss)
        return loss

    def on_after_backward(self) -> None:
        """Log info about params and gradients"""
        if not self.extra_logging:
            return

        trainer = self.trainer

        cats = ["weight", "bias"]
        if trainer.global_step % trainer.log_every_n_steps == 0:
            tensorboard = self.logger.experiment
            for name, parameter in self.named_parameters():
                # log param avg, param hist, grad avg and grad hist
                cat_suffix = cats[next(i for i, cat in enumerate(cats) if cat in name)]
                tensorboard.add_scalar(
                    f"param_avg.{cat_suffix}/{name}",
                    torch.mean(parameter),
                    trainer.global_step,
                )
                tensorboard.add_histogram(
                    f"params.{cat_suffix}/{name}", parameter, trainer.global_step
                )
                tensorboard.add_scalar(
                    f"grad_avg.{cat_suffix}/{name}",
                    torch.mean(parameter.grad),
                    trainer.global_step,
                )
                tensorboard.add_histogram(
                    f"grads.{cat_suffix}/{name}", parameter.grad, trainer.global_step
                )

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        input, target = batch
        logits = self(input)
        loss = F.cross_entropy(logits, target)
        self.log_dict(
            {
                "val/loss": loss,
                **self.val_metrics(logits, target),
            }
        )

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        # breakpoint()
        input, target = batch
        bs, ncrops, c, h, w = input.size()
        logits = self(input.view(-1, c, h, w))
        logits_avg = logits.view(bs, ncrops, -1).mean(1)
        self.log_dict(self.test_metrics(logits_avg, target))

    def configure_optimizers(self) -> dict[str, tp.Any]:
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_opts)
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.1,
                    patience=5,
                    min_lr=self.optimizer_opts["lr"] * 0.1**3,
                ),
                interval="epoch",
                monitor="val/error@1",
            ),
        )
