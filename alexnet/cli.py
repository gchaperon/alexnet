import typing as tp
import pytorch_lightning as pl
import click

import alexnet.models as models
import alexnet.data as data


# Kinda ugly, but DRY is more important I guess
_TaskT = tp.Literal[
    "mnist", "fashion-mnist", "cifar10", "cifar100", "tiny-imagenet", "imagenet"
]
_task_dispatch = dict(
    zip(
        tp.get_args(_TaskT),
        [
            data.LitMNIST,
            data.LitFashionMNIST,
            data.LitCIFAR10,
            data.LitCIFAR100,
            data.LitTinyImageNet,
            data.LitImageNet,
        ],
    )
)


@click.command()
@click.option("--batch-size", default=128)
@click.option("--dropout", default=0.5)
@click.option("--task", type=click.Choice(tp.get_args(_TaskT)), required=True)
@click.option("--learn-rate", default=0.0001)
@click.option("--seed", default=42)
@click.option(
    "--extra-logging",
    is_flag=True,
    default=False,
    help="Wether to log histograms of parameters and grads",
)
def cli(
    batch_size: int,
    dropout: float,
    task: _TaskT,
    learn_rate: float,
    seed: int,
    extra_logging: bool,
) -> None:
    print(f"train called with args {locals()}")
    pl.seed_everything(seed, workers=True)
    datamodule_cls = _task_dispatch[task]
    model = models.LitAlexNet(
        nclasses=datamodule_cls.nclasses,
        optimizer_opts=dict(lr=learn_rate),
        dropout=dropout,
        extra_logging=extra_logging,
    )
    # print(model)
    datamodule = datamodule_cls("data", batch_size=batch_size)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        # deterministic="warn",
        max_epochs=100,
        logger=pl.loggers.TensorBoardLogger(
            save_dir="logs", name=task, default_hp_metric=False, log_graph=True
        ),
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val/error@1",
                # NOTE: val_check_interval is 4 times per epoch, so this
                # patience value means no improvement in the last 10 epochs
                patience=10 * 4,
                mode="min",
                stopping_threshold=0.0,
            ),
            pl.callbacks.LearningRateMonitor(),
        ],
        enable_checkpointing=False,
        val_check_interval=1 / 4,
    )
    trainer.fit(model, datamodule)
    # trainer.test(model, datamodule)
