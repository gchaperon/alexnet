import pytorch_lightning as pl
import click

import alexnet.models as models
import alexnet.data as data


@click.command()
def cli() -> None:
    model = models.LitAlexNet(
        nclasses=data.ImageNet.nclasses,
        optimizer_opts=dict(lr=0.01, momentum=0.9, weight_decay=0.0005),
        dropout=0.5,
    )
    datamodule = data.LitImageNet("data", batch_size=128)

    trainer = pl.Trainer(accelerator="gpu", devices=1)
    trainer.fit(model, datamodule)
