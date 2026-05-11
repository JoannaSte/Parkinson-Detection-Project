from pathlib import Path

from create_dataset import LiderDatasetChunked


class CreateDataloader:
    def __init__(
        self,
        annotations_file: Path,
        audio_directory: Path,
        task_number: int,
        fold: int,
        processor: str,
        sample_rate: int,
        num_samples: int,
        device: str = "cpu",
        overlap: float = 0.0,
        chunk_index: int | None = None,
        skip_too_short: bool = True,
    ):
        self.annotations_file = annotations_file
        self.audio_directory = audio_directory
        self.task_number = task_number
        self.fold = fold
        self.processor = processor
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.device = device
        self.overlap = overlap
        self.chunk_index = chunk_index
        self.skip_too_short = skip_too_short

    def _make_dataset(self, eval_or_train: str) -> LiderDatasetChunked:
        return LiderDatasetChunked(
            audio_dir_file=self.audio_directory,
            annotations_file=self.annotations_file,
            task_number=self.task_number,
            target_sample_rate=self.sample_rate,
            num_samples=self.num_samples,
            device=self.device,
            fold=self.fold,
            eval_or_train=eval_or_train,
            processor=self.processor,
            overlap=self.overlap,
            chunk_index=self.chunk_index,
            skip_too_short=self.skip_too_short,
        )

    def create_eval_dataset(self) -> LiderDatasetChunked:
        return self._make_dataset("eval")

    def create_train_dataset(self) -> LiderDatasetChunked:
        return self._make_dataset("train")
