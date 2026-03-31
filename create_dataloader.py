from pathlib import Path
from torch.utils.data import ConcatDataset

from create_dataset import MyOwnData


class CreateDataloader:
    def __init__(
        self,
        train_file: Path,
        eval_file: Path,
        audio_directory: Path,
        sample_rate: int,
        num_samples: int,
        device: str = "cpu",
        augment: bool = False,
        aug_params: dict | None = None,
    ):
        self.train_file = train_file
        self.eval_file = eval_file
        self.audio_directory = audio_directory

        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.device = device

        self.augment = augment
        self.aug_params = aug_params or {}

    def create_eval_dataset(self):
        return MyOwnData(
            annotations_file=self.eval_file,
            audio_dir=self.audio_directory,
            target_sample_rate=self.sample_rate,
            num_samples=self.num_samples,
            device=self.device,
            augment=False,  
        )

    def create_train_dataset(self):
        base_dataset = MyOwnData(
            annotations_file=self.train_file,
            audio_dir=self.audio_directory,
            target_sample_rate=self.sample_rate,
            num_samples=self.num_samples,
            device=self.device,
            augment=False,
        )

        if not self.augment:
            return base_dataset

        augmented_dataset = MyOwnData(
            annotations_file=self.train_file,
            audio_dir=self.audio_directory,
            target_sample_rate=self.sample_rate,
            num_samples=self.num_samples,
            device=self.device,
            augment=True,
            aug_params=self.aug_params,
        )

        return ConcatDataset([base_dataset, augmented_dataset])