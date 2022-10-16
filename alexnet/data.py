import PIL.Image
import xml.etree.ElementTree as ElementTree
import joblib
import tqdm
import pathlib
import typing as tp
import PIL.Image
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
import numpy as np
import numpy.typing as npt


class ImageNetItem(tp.NamedTuple):
    image: torch.Tensor
    cls: str


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


T = tp.TypeVar("T")


class ImageNet(torch.utils.data.Dataset[T]):
    name = "imagenet"
    nclasses = 1000

    datadir: pathlib.Path
    split: tp.Literal["train", "val", "test"]
    image_paths: npt.NDArray[np.str_]
    wnid_to_index: tp.Dict[str, int]
    wnid_to_name: tp.Dict[str, str]
    image_transforms: tp.Callable[[PIL.Image.Image], torch.Tensor]

    @tp.overload
    def __init__(
        self: "ImageNet[ImageNetItem]", datadir: str, split: tp.Literal["train", "val"]
    ) -> None:
        ...

    @tp.overload
    def __init__(
        self: "ImageNet[torch.Tensor]", datadir: str, split: tp.Literal["test"]
    ) -> None:
        ...

    def __init__(
        self,
        datadir: str,
        split: tp.Literal["train", "val", "test"],
    ) -> None:
        super().__init__()
        self.datadir = pathlib.Path(datadir)
        self.split = split

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

        self.image_transforms = lambda: None

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

    def __getitem__(self, key: int) -> T:
        image_path = pathlib.Path(str(self.image_paths[key]))
        image = PIL.Image.open(image_path).convert("RGB")
        if self.split == "test":
            return tp.cast(T, image)
        else:
            wnid = self.get_wnid(image_path)
            return tp.cast(T, ImageNetItem(image, wnid))

    def __len__(self) -> int:
        return len(self.image_paths)


class LitTinyImageNet(pl.LightningDataModule):
    _val_ratio: tp.ClassVar[float] = 0.1

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
