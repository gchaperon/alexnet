import tqdm
import pathlib
import typing as tp
import PIL.Image
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms


class ImageNetItem(tp.NamedTuple):
    image: torch.Tensor
    cls: tp.Optional[int]


class TinyImageNet(torch.utils.data.Dataset[ImageNetItem]):
    nclasses = 200

    datadir: pathlib.Path
    split: tp.Literal["train", "val", "test"]
    transform: tp.Callable

    wnid_to_idx: tp.Dict[str, int]
    wnid_to_name: tp.Dict[str, str]
    image_data: tp.List[tp.Tuple[pathlib.Path, tp.Optional[str]]]

    def __init__(
        self,
        datadir: str,
        split: tp.Literal["train", "val", "test"],
        transform: tp.Optional[tp.Callable] = None,
    ) -> None:
        super().__init__()
        self.datadir = pathlib.Path(datadir)
        self.split = split
        self.transform = transform or transforms.ToTensor()

        with open(self.datadir / "tiny-imagenet" / "wnids.txt") as wnids_file:
            self.wnid_to_idx = {line.strip(): i for i, line in enumerate(wnids_file)}
        with open(self.datadir / "tiny-imagenet" / "words.txt") as words_file:
            self.wnid_to_name = dict(line.strip().split("\t") for line in words_file)

        image_iter = (self.datadir / "tiny-imagenet" / split).glob("**/*.JPEG")
        if split == "train":
            self.image_data = [
                (image_path, image_path.parts[-3]) for image_path in image_iter
            ]
        elif split == "val":
            with open(
                self.datadir / "tiny-imagenet" / "val" / "val_annotations.txt"
            ) as ann_file:
                image_name_to_annotation = dict(line.split()[:2] for line in ann_file)
            self.image_data = [
                (image_path, image_name_to_annotation[image_path.name])
                for image_path in image_iter
            ]
        elif split == "test":
            self.image_data = [(image_path, None) for image_path in image_iter]

    def __getitem__(self, key: int) -> ImageNetItem:
        image_path, wnid = self.image_data[key]
        idx = self.wnid_to_idx[wnid] if wnid else None
        return ImageNetItem(
            self.transform(PIL.Image.open(image_path).convert("RGB")),
            idx,
        )

    def __len__(self) -> int:
        return len(self.image_data)


class LitTinyImageNet(pl.LightningDataModule):
    _val_ratio = 0.1
    datadir: str
    batch_size: int

    def __init__(self, datadir: str, batch_size: int) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("datadir",))

        self.datadir = datadir
        self.batch_size = batch_size

    def setup(
        self, stage: tp.Optional[tp.Literal["fit", "validate", "test"]] = None
    ) -> None:
        if stage in ("fit", "validate", None):
            full_train = TinyImageNet(self.datadir, split="train")
            val_len = int(len(full_train) * self._val_ratio)
            train_len = len(full_train) - val_len
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                dataset=full_train, lengths=[train_len, val_len]
            )
        if stage in ("test", None):
            self.test_dataset = TinyImageNet(self.datadir, split="val")
        if stage in ("predict", None):
            self.predict_dataset = TinyImageNet(self.datadir, split="test")

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )
