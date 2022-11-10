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
import torchvision

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


class ImageNet(torch.utils.data.Dataset[ImageNetItem]):
    name = "imagenet"
    nclasses = 1000

    datadir: pathlib.Path
    split: tp.Literal["train", "val"]
    image_paths: npt.NDArray[np.str_]
    wnid_to_index: tp.Dict[str, int]
    wnid_to_name: tp.Dict[str, str]
    image_transform: _TransformT

    def __init__(
        self,
        datadir: str,
        split: tp.Literal["train", "val"],
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

    def __getitem__(self, key: int) -> ImageNetItem:
        image_path = pathlib.Path(str(self.image_paths[key]))
        image = PIL.Image.open(image_path).convert("RGB")
        transformed = self.image_transform(image)
        wnid = self.get_wnid(image_path)
        target = torch.tensor(self.wnid_to_index[wnid])
        return ImageNetItem(transformed, target)

    def __len__(self) -> int:
        return len(self.image_paths)


class TinyImageNet(torch.utils.data.Dataset[ImageNetItem]):
    name = "tiny-imagenet-200"
    val_annotations: tp.Optional[dict[str, str]]

    def __init__(
        self,
        datadir: str,
        split: tp.Literal["train", "val"],
        transform: tp.Optional[_TransformT] = None,
    ) -> None:
        super().__init__()
        self.datadir = pathlib.Path(datadir)
        self.split = split
        self.image_transform = transform or (lambda image: image)

        memory = joblib.Memory(datadir, verbose=0)
        self.image_paths = memory.cache(find_files)(
            self.datadir / self.name / self.split,
            extension="JPEG",
        )
        with open(self.datadir / self.name / "wnids.txt") as wnids_file:
            self.wnid_to_index = {line.strip(): i for i, line in enumerate(wnids_file)}

        self.wnid_to_name = {}
        with open(self.datadir / self.name / "words.txt") as names_file:
            for line in names_file:
                wnid, name = line.strip().split("\t")
                self.wnid_to_name[wnid] = name

        self.val_annotations = None
        if self.split == "val":
            self.val_annotations = {}
            with open(
                self.datadir / self.name / "val" / "val_annotations.txt"
            ) as val_ann_file:
                for line in val_ann_file:
                    img_name, ann, *_ = line.split("\t")
                    self.val_annotations[img_name] = ann

    def _get_wnid(self, image_path: pathlib.Path) -> str:
        if self.split == "val":
            assert self.val_annotations is not None
            return self.val_annotations[image_path.name]
        elif self.split == "train":
            return image_path.parts[-3]

    def __getitem__(self, key: int) -> ImageNetItem:
        image_path = pathlib.Path(str(self.image_paths[key]))
        image = PIL.Image.open(image_path).convert("RGB")
        transformed = self.image_transform(image)
        wnid = self._get_wnid(image_path)
        target = self.wnid_to_index[wnid]
        return ImageNetItem(transformed, torch.tensor(target))

    def __len__(self) -> int:
        return len(self.image_paths)


class _NormalizeArgsT(tp.TypedDict):
    mean: tp.List[float]
    std: tp.List[float]


class _BaseDataModule(pl.LightningDataModule):
    # These should be defined in setup()
    train_dataset: torch.utils.data.Dataset[ImageNetItem]
    val_dataset: torch.utils.data.Dataset[ImageNetItem]
    test_dataset: torch.utils.data.Dataset[ImageNetItem]

    # Classvars should be defined by inheriting classes
    nclasses: tp.ClassVar[int]
    dataset_cls: "???? I dunno how to type this"
    _total_train: tp.ClassVar[int]
    _nval: tp.ClassVar[int]
    _normalize_args: tp.ClassVar[_NormalizeArgsT]

    # Should be defined in __init__ by subclasses
    train_transform: tp.Callable[[PIL.Image.Image], torch.Tensor]
    val_transform: tp.Callable[[PIL.Image.Image], torch.Tensor]
    test_transform: tp.Callable[[PIL.Image.Image], torch.Tensor]

    # Defined in __init__
    datadir: str
    batch_size: int

    def __init__(self, datadir: str, batch_size: int) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("datadir",))

        self.datadir = datadir
        self.batch_size = batch_size

    def train_dataloader(self) -> torch.utils.data.DataLoader[ImageNetItem]:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=min(os.cpu_count() or 0, 8),
            drop_last=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader[ImageNetItem]:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=2 * self.batch_size,
            shuffle=False,
            num_workers=min(os.cpu_count() or 0, 8),
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader[ImageNetItem]:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            # NOTE: Each batch will contain 5 or 10 images because of 5 or 10
            # crop, so these batches use a lot of memory
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=min(os.cpu_count() or 0, 8),
        )


class _BaseTorchVisionDataModule(_BaseDataModule):
    def prepare_data(self) -> None:
        self.dataset_cls(self.datadir, train=True, download=True)
        self.dataset_cls(self.datadir, train=False, download=True)

    def setup(
        self, stage: tp.Optional[tp.Literal["fit", "validate", "test"]] = None
    ) -> None:
        if stage in ("fit", "validate", None):
            # FIXME: this is wrong since fit and validate are called two
            # separate times and each time produces a different random perm
            #
            # or is it?
            indices = torch.randperm(self._total_train).tolist()

            self.val_dataset = torch.utils.data.Subset(
                self.dataset_cls(
                    self.datadir, train=True, transform=self.val_transform
                ),
                indices=indices[: self._nval],
            )
            self.train_dataset = torch.utils.data.Subset(
                self.dataset_cls(
                    self.datadir, train=True, transform=self.train_transform
                ),
                indices=indices[self._nval :],
            )

        if stage in ("test", None):
            self.test_dataset = self.dataset_cls(
                self.datadir, train=False, transform=self.test_transform
            )


class LitMNIST(_BaseTorchVisionDataModule):
    nclasses = 10
    dataset_cls = torchvision.datasets.MNIST
    _total_train = 60_000
    _nval = 3000
    _normalize_args = dict(mean=[0.1307] * 3, std=[0.3081] * 3)

    def __init__(self, datadir: str, batch_size: int) -> None:
        super().__init__(datadir, batch_size)

        self.train_transform = transforms.Compose(
            [
                transforms.ToRGB(),
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(**self._normalize_args),
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.ToRGB(),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(**self._normalize_args),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToRGB(),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(**self._normalize_args),
                transforms.Unsqueeze(0),
            ]
        )


class LitFashionMNIST(_BaseTorchVisionDataModule):
    nclasses: tp.ClassVar[int] = 10
    dataset_cls = torchvision.datasets.FashionMNIST
    _total_train = 60_000
    _nval = 3000
    _normalize_args = dict(mean=[0.286] * 3, std=[0.353] * 3)

    def __init__(self, datadir: str, batch_size: int) -> None:
        super().__init__(datadir, batch_size)

        self.train_transform = transforms.Compose(
            [
                transforms.ToRGB(),
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**self._normalize_args),
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.ToRGB(),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(**self._normalize_args),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToRGB(),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(**self._normalize_args),
                transforms.Unsqueeze(0),
            ]
        )


class LitCIFAR10(_BaseTorchVisionDataModule):
    nclasses = 10
    dataset_cls = torchvision.datasets.CIFAR10
    _total_train = 50_000
    _nval = 2000
    _normalize_args = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])

    def __init__(self, datadir: str, batch_size: int) -> None:
        super().__init__(datadir, batch_size)

        self.train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.PCAAugment(),
                transforms.Normalize(**self._normalize_args),
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(**self._normalize_args),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(**self._normalize_args),
                transforms.TenCrop(224),
                transforms.Stack(),
            ]
        )


class LitCIFAR100(LitCIFAR10):
    nclasses = 100
    dataset_cls = torchvision.datasets.CIFAR100
    _total_train = 50_000
    _nval = 2000
    _normalize_args = dict(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    # NOTE: CIFAR100 uses the exact same transforms as CIFAR10


class _BaseImageNetDataModule(_BaseDataModule):
    def setup(
        self, stage: tp.Optional[tp.Literal["fit", "validate", "test"]] = None
    ) -> None:
        if stage in ("fit", "validate", None):
            indices = torch.randperm(self._total_train).tolist()
            self.val_dataset = torch.utils.data.Subset(
                self.dataset_cls(
                    self.datadir, split="train", transform=self.val_transform
                ),
                indices=indices[: self._nval],
            )
            self.train_dataset = torch.utils.data.Subset(
                self.dataset_cls(
                    self.datadir, split="train", transform=self.train_transform
                ),
                indices=indices[self._nval :],
            )
        if stage in ("test", None):
            self.test_dataset = self.dataset_cls(
                self.datadir, split="val", transform=self.test_transform
            )


class LitImageNet(_BaseImageNetDataModule):
    """ImageNet data module specifically for training AlexNet.
    Data should already been downloaded, it requires signing a user agreement."""

    nclasses: int = 1000
    dataset_cls = ImageNet
    _total_train = 1_281_167
    _nval = 50_000
    _normalize_args = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __init__(self, datadir: str, batch_size: int) -> None:
        super().__init__(datadir, batch_size)
        self.train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.PCAAugment(),
                transforms.Normalize(**self._normalize_args),
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(**self._normalize_args),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(**self._normalize_args),
                transforms.TenCrop(224),
                transforms.Stack(),
            ]
        )


class LitTinyImageNet(_BaseImageNetDataModule):
    nclasses = 200
    dataset_cls = TinyImageNet
    _total_train = 100_000
    _nval = 5000
    _normalize_args = dict(mean=[0.4802, 0.4481, 0.3975], std=[0.2764, 0.2689, 0.2816])

    def __init__(self, datadir: str, batch_size: int) -> None:
        super().__init__(datadir, batch_size)

        self.train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.PCAAugment(),
                transforms.Normalize(**self._normalize_args),
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(**self._normalize_args),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(**self._normalize_args),
                transforms.TenCrop(224),
                transforms.Stack(),
            ]
        )
