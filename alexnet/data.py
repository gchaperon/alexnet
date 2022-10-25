import os
import xml.etree.ElementTree as ElementTree
import joblib
import tqdm
import pathlib
import typing as tp
import PIL.Image
import pytorch_lightning as pl
import torch
import numpy as np
import numpy.typing as npt

from . import transforms

_ImageT = tp.Union[torch.Tensor, PIL.Image.Image]
_TransformT = tp.Callable[[PIL.Image.Image], _ImageT]


class ImageNetItem(tp.NamedTuple):
    image: _ImageT
    target: torch.Tensor


def find_files(root_dir: pathlib.Path, extension: str) -> npt.NDArray[np.str_]:
    return np.stack(
        [
            np.array(str(path), dtype=np.str_)
            for path in tqdm.tqdm(
                root_dir.glob(f"**/*.{extension}"),
                desc=f"Searching {root_dir} for .{extension} files",
            )
        ]
    )


def _identity_transform(image: PIL.Image.Image) -> PIL.Image.Image:
    return image


class ImageNet(torch.utils.data.Dataset[tp.Union[ImageNetItem, _ImageT]]):
    name = "imagenet"
    nclasses = 1000

    datadir: pathlib.Path
    split: tp.Literal["train", "val", "test"]
    image_paths: npt.NDArray[np.str_]
    wnid_to_index: tp.Dict[str, int]
    wnid_to_name: tp.Dict[str, str]
    image_transform: _TransformT

    def __init__(
        self,
        datadir: str,
        split: tp.Literal["train", "val", "test"],
        transform: tp.Optional[_TransformT] = None,
    ) -> None:
        super().__init__()
        self.datadir = pathlib.Path(datadir)
        self.split = split
        self.image_transform = transform or (lambda image: image)

        memory = joblib.Memory(datadir, verbose=0)
        self.image_paths = memory.cache(find_files)(
            self.datadir / self.name / "ILSVRC" / "Data" / "CLS-LOC" / self.split,
            extension="JPEG",
        )

        self.wnid_to_index = {}
        self.wnid_to_name = {}
        with open(self.datadir / self.name / "LOC_synset_mapping.txt") as mapping_file:
            for index, line in enumerate(mapping_file):
                wnid, _, name = line.strip().partition(" ")
                self.wnid_to_index[wnid] = index
                self.wnid_to_name[wnid] = name

    @staticmethod
    def to_ann_path(image_path: pathlib.Path) -> pathlib.Path:
        """Only val images are guaranteed to have a relative ann file"""
        parts = list(image_path.parts)
        parts[parts.index("Data")] = "Annotations"
        return pathlib.Path(*parts).with_suffix(".xml")

    def get_wnid(self, image_path: pathlib.Path) -> str:
        if self.split == "train":
            return image_path.parent.name
        elif self.split == "val":
            ann_path = self.to_ann_path(image_path)
            element = ElementTree.parse(ann_path).getroot().find("./object/name")
            text = element.text if element is not None else ""
            if text:
                return text
            else:
                raise ValueError(f"wnid not found in {ann_path=}")
        else:
            raise ValueError("only train and test splits have targets")

    def __getitem__(self, key: int) -> tp.Union[ImageNetItem, _ImageT]:
        image_path = pathlib.Path(str(self.image_paths[key]))
        image = PIL.Image.open(image_path).convert("RGB")
        transformed = self.image_transform(image)
        if self.split == "test":
            return transformed
        else:
            wnid = self.get_wnid(image_path)
            target = torch.tensor(self.wnid_to_index[wnid])
            return ImageNetItem(transformed, target)

    def __len__(self) -> int:
        return len(self.image_paths)


_default_train_transform: tp.Callable[
    [PIL.Image.Image], torch.Tensor
] = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.PCAAugment(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

_default_val_transform: tp.Callable[
    [PIL.Image.Image], torch.Tensor
] = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

_default_test_transform: tp.Callable[
    [PIL.Image.Image], torch.Tensor
] = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.TenCrop(224),
        transforms.Stack(),
    ]
)


# NOTE: types in {train, val, ...}_dataloader() methods are a mess and i'm
# kinda tired so i'm just silencing them
class LitImageNet(pl.LightningDataModule):
    """ImageNet data module specifically for training AlexNet"""

    _val_ratio: tp.ClassVar[float] = 0.1

    datadir: str
    batch_size: int
    train_transform: _TransformT
    val_transform: _TransformT
    test_transform: _TransformT

    def __init__(
        self,
        datadir: str,
        batch_size: int,
        train_transform: _TransformT = _default_train_transform,
        val_transform: _TransformT = _default_val_transform,
        test_transform: _TransformT = _default_test_transform,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("datadir",))

        self.datadir = datadir
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(
        self, stage: tp.Optional[tp.Literal["fit", "validate", "test"]] = None
    ) -> None:
        if stage in ("fit", "validate", None):
            # NOTE: for val stage use the val dataset but using a simpler
            # transform, where the data is simply an image instead of
            # classifying a ten crop and averaging.
            self.val_dataset = ImageNet(
                self.datadir, split="val", transform=self.val_transform
            )
        if stage in ("fit", None):
            self.train_dataset = ImageNet(
                self.datadir, split="train", transform=self.train_transform
            )
        if stage in ("test", None):
            # NOTE: for test stage also use val dataset since actual test
            # dataset doesn't have targets. However, use the actual test
            # transform.
            self.test_dataset = ImageNet(
                self.datadir, split="val", transform=self.test_transform
            )
        if stage in ("predict", None):
            self.predict_dataset = ImageNet(
                self.datadir, split="test", transform=self.test_transform
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader[ImageNetItem]:
        return torch.utils.data.DataLoader(
            self.train_dataset,  # type:ignore
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=min(os.cpu_count() or 0, 8),
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader[ImageNetItem]:
        return torch.utils.data.DataLoader(
            self.val_dataset,  # type:ignore
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=min(os.cpu_count() or 0, 8),
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader[ImageNetItem]:
        return torch.utils.data.DataLoader(
            self.test_dataset,  # type:ignore
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=min(os.cpu_count() or 0, 8),
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader[torch.Tensor]:
        return torch.utils.data.DataLoader(
            self.predict_dataset,  # type:ignore
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=min(os.cpu_count() or 0, 8),
        )
