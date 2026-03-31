import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from audiomentations import (
    LowPassFilter,
    AddGaussianNoise,
    TimeStretch,
    PitchShift,
)

class MyOwnData(Dataset):
    def __init__(
        self,
        annotations_file: str,
        audio_dir: str,
        target_sample_rate: int,
        num_samples: int,
        device: str = "cpu",
        augment: bool = False,
        aug_params: dict | None = None,
    ):
        self.audio_dir = audio_dir
        self.annotations = pd.read_csv(annotations_file)

        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.device = device

        self.augment = augment
        self.aug_params = aug_params or {}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        row = self.annotations.iloc[index]

        audio_path = self._get_audio_sample_path(row)
        label = row["label"]
        sample_id = self._get_id(row)

        signal, sr = torchaudio.load(audio_path)

        signal = self._resample(signal, sr)
        signal = self._trim_or_pad(signal)

        if self.augment:
            signal, sample_id = self._apply_augmentation(signal, sample_id)

        return {
            "input_values": signal.squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
            "id": sample_id,
        }


    def _get_audio_sample_path(self, row):
        return os.path.join(self.audio_dir, row["fold"], row["filename"])

    def _get_id(self, row):
        if "id" in row:
            return str(row["id"])
        return os.path.splitext(row["filename"])[0]


    def _resample(self, signal: torch.Tensor, sr: int):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _trim_or_pad(self, signal: torch.Tensor):
        length = signal.shape[1]

        if length > self.num_samples:
            signal = signal[:, : self.num_samples]
        else:
            padding = self.num_samples - length
            signal = torch.nn.functional.pad(signal, (0, padding))

        return signal


    def _apply_augmentation(self, signal: torch.Tensor, sample_id: str):
        signal_np = signal.squeeze().cpu().numpy()

        augmentations = []


        if "low_pass" in self.aug_params:
            params = self.aug_params["low_pass"]
            augmentations.append(
                LowPassFilter(
                    min_cutoff_freq=params["min"],
                    max_cutoff_freq=params["max"],
                    p=params.get("p", 0.5),
                )
            )

        if "noise" in self.aug_params:
            params = self.aug_params["noise"]
            augmentations.append(
                AddGaussianNoise(
                    min_amplitude=params["min"],
                    max_amplitude=params["max"],
                    p=params.get("p", 0.5),
                )
            )

        if "time_stretch" in self.aug_params:
            params = self.aug_params["time_stretch"]
            augmentations.append(
                TimeStretch(
                    min_rate=params["min"],
                    max_rate=params["max"],
                    p=params.get("p", 0.5),
                )
            )

        if "pitch_shift" in self.aug_params:
            params = self.aug_params["pitch_shift"]
            augmentations.append(
                PitchShift(
                    min_semitones=params["min"],
                    max_semitones=params["max"],
                    p=params.get("p", 0.5),
                )
            )

        for aug in augmentations:
            signal_np = aug(signal_np, sample_rate=self.target_sample_rate)

        signal_tensor = torch.tensor(signal_np, dtype=torch.float32).unsqueeze(0)

        return signal_tensor, sample_id + "_aug"