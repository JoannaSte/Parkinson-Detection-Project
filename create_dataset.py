import math
import os
import random
import librosa
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor


class LiderDatasetChunked(Dataset):
    def __init__(self,
                 audio_dir_file,
                 annotations_file,
                 task_number,
                 target_sample_rate,
                 num_samples,
                 device,
                 fold,
                 eval_or_train,
                 processor,
                 overlap=0.0,
                 chunk_index=None,
                 skip_too_short=True):
        """
        chunk_index:
            None  -> wszystkie chunki z każdego nagrania (jak wcześniej)
            int   -> tylko ten konkretny chunk (0 = 0-2s, 1 = 2-4s, ...)
                     wspiera negatywne indeksy (-1 = ostatni chunk)
        skip_too_short:
            True  -> pomija nagrania zbyt krótkie na żądany chunk
            False -> dodaje chunk z paddingiem zerami
        """
        self.audio_dir_file = audio_dir_file
        self.device = device
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.length_sec = num_samples / target_sample_rate
        self.overlap = overlap
        self.chunk_index = chunk_index
        self.skip_too_short = skip_too_short

        # Wczytaj i przefiltruj anotacje
        annotations = pd.read_csv(annotations_file)
        annotations = annotations[annotations['experiment_number'] == task_number]
        if eval_or_train == 'eval':
            annotations = annotations[annotations['fold'] == fold]
        else:
            annotations = annotations[annotations['fold'] != fold]
        self.annotations = annotations.reset_index(drop=True)

        # Procesor ładowany RAZ
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(processor)

        # Wygeneruj indeks chunków
        self.chunks = self._build_chunk_index()

    def _build_chunk_index(self):
        """Buduje listę [(row_idx, start_sec), ...] zależnie od chunk_index."""
        chunks = []
        step_sec = self.length_sec * (1 - self.overlap)

        for idx, row in self.annotations.iterrows():
            duration = row['duration']

            # Wszystkie możliwe chunki dla tego nagrania
            if duration <= self.length_sec:
                all_starts = [0.0]
            else:
                n_chunks = int(math.floor((duration - self.length_sec) / step_sec)) + 1
                all_starts = [i * step_sec for i in range(n_chunks)]

            # Wybór: wszystkie czy konkretny chunk?
            if self.chunk_index is None:
                # Wszystkie chunki
                for start in all_starts:
                    chunks.append((idx, start))
            else:
                # Konkretny chunk
                try:
                    chosen_start = all_starts[self.chunk_index]
                    chunks.append((idx, chosen_start))
                except IndexError:
                    # Nagranie za krótkie na żądany chunk
                    if self.skip_too_short:
                        continue  # pomijamy
                    else:
                        # Padding: bierzemy start = 0 i tak, że i tak będzie zerami dopełnione
                        chunks.append((idx, 0.0))

        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        row_idx, start_sec = self.chunks[index]
        row = self.annotations.iloc[row_idx]

        audio_path = self._get_audio_path(row)
        signal, _ = librosa.load(audio_path, sr=self.target_sample_rate)

        processed = self.feature_extractor(
            signal, sampling_rate=self.target_sample_rate, return_tensors="pt"
        )
        signal = processed['input_values'][0]

        start = int(start_sec * self.target_sample_rate)
        stop = start + self.num_samples
        signal = signal[start:stop]

        if signal.size(0) < self.num_samples:
            signal = self._right_pad(signal)

        signal = signal.to(self.device)

        return {
            'input_values': signal,
            'label': row['label'],
            'id': row['name'],
            'chunk_start': start_sec,
        }

    def _get_audio_path(self, row):
        path = row['path'].replace("\\", "/")
        return os.path.join(self.audio_dir_file, path)

    def _right_pad(self, signal):
        missing = self.num_samples - signal.size(0)
        return torch.nn.functional.pad(signal, (0, missing))