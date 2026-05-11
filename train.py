import yaml
import torch
import lightning as pl
from pathlib import Path
from fire import Fire

from Trainer import Learner
from create_dataloader import CreateDataloader

from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_callbacks(output_dir: Path):
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="model-{epoch:02d}-{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    return [checkpoint_callback, lr_monitor]


def build_logger(output_dir: Path):
    return TensorBoardLogger(
        save_dir=str(output_dir / "logs"),
        name="wavlm_parkinson_detection",
    )


def main(config_file: str):
    cfg = load_config(config_file)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"""
        ===== Experiment Config =====
        Dataset: {cfg['dataset']}
        Task: {cfg['task_number']}
        Fold: {cfg['fold']}
        Processor: {cfg['processor']}
        =============================
        """
    )

    data_builder = CreateDataloader(
        annotations_file=cfg["annotations_file"],
        audio_directory=cfg["audio_directory"],
        task_number=cfg["task_number"],
        fold=cfg["fold"],
        processor=cfg["processor"],
        sample_rate=cfg["sample_rate"],
        num_samples=cfg["num_samples"],
        device=cfg.get("device", "cpu"),
        overlap=cfg.get("overlap", 0.0),
        chunk_index=cfg.get("chunk_index", None),
        skip_too_short=cfg.get("skip_too_short", True),
    )

    train_dataset = data_builder.create_train_dataset()
    val_dataset = data_builder.create_eval_dataset()

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )

    learner = Learner(
        model=cfg["model"],  
        loss_fn=torch.nn.CrossEntropyLoss(),
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )


    logger = build_logger(output_dir)
    callbacks = build_callbacks(output_dir)


    trainer = pl.Trainer(
        max_epochs=cfg.get("max_epochs", 50),
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        log_every_n_steps=10,
        default_root_dir=str(output_dir),
        callbacks=callbacks,
        logger=logger,
    )


    trainer.fit(
        model=learner,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


    trainer.validate(
        model=learner,
        dataloaders=val_loader,
        ckpt_path="best",
    )


if __name__ == "__main__":
    Fire(main)