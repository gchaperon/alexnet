import typing as tp
import pytorch_lightning as pl
import click

import alexnet.models as models
import alexnet.data as data


_task_dispatch = {
    "mnist": data.LitMNIST,
    "fashionmnist": data.LitFashionMNIST,
    "cifar10": data.LitCIFAR10,
    "cifar100": data.LitCIFAR100,
}


@click.command()
@click.option("--batch-size", default=128)
@click.option("--dropout", default=0.5)
@click.option("--task", type=click.Choice(list(_task_dispatch.keys())), required=True)
def cli(
    batch_size: int,
    dropout: float,
    task: tp.Literal["mnist", "fashionmnist", "cifar10", "cifar100"],
) -> None:
    pl.seed_everything(42, workers=True)
    datamodule_cls = _task_dispatch[task]
    model = models.LitAlexNet(
        nclasses=datamodule_cls.nclasses,
        # optimizer_opts=dict(lr=0.01, momentum=0.9, weight_decay=0.0005),
        optimizer_opts=dict(lr=0.0001),
        dropout=dropout,
    )
    # print(model)
    datamodule = datamodule_cls("data", batch_size=batch_size)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        # deterministic="warn",
        max_epochs=100,
        logger=pl.loggers.TensorBoardLogger(
            save_dir="logs", name=task, default_hp_metric=False
        ),
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val/error@1",
                patience=5 * 4,
                mode="min",
                stopping_threshold=0.0,
            ),
            pl.callbacks.LearningRateMonitor(),
        ],
        enable_checkpointing=False,
        val_check_interval=1 / 4,
    )
    trainer.fit(model, datamodule)
