import csv
import random
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader


CLIP_LENGTH = 16
CROP_SIZE = 64
NUM_CLASSES = 101


# Charger les lignes du manifeste UCF101
def load_manifest(
    manifest_path: Union[str, Path] = "manifest.csv",
):
    with Path(manifest_path).open(
        newline="",
        encoding="utf-8",
    ) as manifest_file:
        return list(csv.DictReader(manifest_file))


# Classe pour appliquer les transformations aux clips vidéo
class TransformedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows,
        train: bool = False,
        clip_index: int = 0,
        transform: Optional[Callable] = None,
    ):
        self.rows = rows
        self.train = train
        self.clip_index = clip_index
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        frames = np.load(row["frames"]).astype(np.float32) / 255.0

        max_start = max(0, len(frames) - CLIP_LENGTH)

        if self.train and max_start > 0:
            start = random.randint(0, max_start)
        elif max_start > 0:
            start_positions = np.linspace(
                0,
                max_start,
                5,
                dtype=int,
            )
            start = int(
                start_positions[min(self.clip_index, 4)]
            )
        else:
            start = 0

        clip = frames[start : start + CLIP_LENGTH]

        if len(clip) < CLIP_LENGTH:
            padding = CLIP_LENGTH - len(clip)
            clip = np.pad(
                clip,
                ((0, padding), (0, 0), (0, 0), (0, 0)),
                mode="edge",
            )

        if self.transform is not None:
            clip = self.transform(clip)

        clip = torch.from_numpy(clip).permute(3, 0, 1, 2)
        target = int(row["label"])

        return clip.contiguous(), target, row["video_id"]


# Transformations pour les clips d'entraînement
def transforms_train(clip):
    _, height, width, _ = clip.shape

    top = random.randint(0, height - CROP_SIZE)
    left = random.randint(0, width - CROP_SIZE)
    clip = clip[:, top : top + CROP_SIZE, left : left + CROP_SIZE]

    if random.random() < 0.5:
        clip = clip[:, :, ::-1].copy()

    return clip


# Transformations pour les clips de validation et de test
def transforms_validation(clip):
    _, height, width, _ = clip.shape
    top = max(0, (height - CROP_SIZE) // 2)
    left = max(0, (width - CROP_SIZE) // 2)
    return clip[:, top : top + CROP_SIZE, left : left + CROP_SIZE]


# Application des transformations aux datasets d'entraînement et de validation
all_rows = load_manifest()
train_rows = [row for row in all_rows if row["split"] == "train"]
validation_rows = [row for row in all_rows if row["split"] == "val"]

train_dataset = TransformedDataset(
    train_rows,
    train=True,
    transform=transforms_train,
)

validation_dataset = TransformedDataset(
    validation_rows,
    train=False,
    transform=transforms_validation,
)

# Données pour l'entraînement
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
)


def get_test_loader(
    manifest_path: Union[str, Path] = "manifest.csv",
    batch_size: int = 32,
    num_workers: int = 4,
    clip_index: int = 0,
):
    rows = [
        row
        for row in load_manifest(manifest_path)
        if row["split"] == "test"
    ]

    test_dataset = TransformedDataset(
        rows,
        train=False,
        clip_index=clip_index,
        transform=transforms_validation,
    )

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
