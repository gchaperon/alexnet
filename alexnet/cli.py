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


@click.command(context_settings=dict(show_default=True))
@click.option("--task", type=click.Choice(tp.get_args(_TaskT)), required=True)
@click.option("--batch-size", default=128)
@click.option("--dropout", default=0.5)
@click.option("--learn-rate", default=0.0001)
@click.option("--seed", default=12331)
@click.option(
    "--extra-logging",
    is_flag=True,
    default=False,
    help="Whether to log histograms of parameters and grads.",
)
@click.option(
    "--fast-dev-run",
    is_flag=True,
    default=False,
    help="Run only a couple of steps, to check if everything is working properly.",
)
def cli(
    task: _TaskT,
    batch_size: int,
    dropout: float,
    learn_rate: float,
    seed: int,
    extra_logging: bool,
    fast_dev_run: bool,
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
    datamodule = datamodule_cls("data", batch_size=batch_size)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/error@1", mode="min"
    )
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
                patience=10,
                mode="min",
                stopping_threshold=0.0,
            ),
            checkpoint_callback,
            pl.callbacks.LearningRateMonitor(),
        ],
        fast_dev_run=fast_dev_run,
    )
    trainer.fit(model, datamodule)
    trainer.test(
        model, datamodule, ckpt_path=checkpoint_callback.best_model_path or None
    )
