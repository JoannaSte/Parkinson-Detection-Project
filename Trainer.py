import torch
import numpy as np
import lightning as pl
from typing import Any, Dict

from sklearn.metrics import accuracy_score, f1_score


class Learner(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # zapisuje hyperparametry do checkpointów
        self.save_hyperparameters(ignore=["model", "loss_fn"])

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        return self.model(input_values)

    def _shared_step(self, batch: Dict[str, torch.Tensor]):
        inputs = batch["input_values"]
        labels = batch["label"]

        outputs = self(inputs)
        logits = outputs.logits

        loss = self.loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=1)

        return loss, preds, labels

    def _compute_metrics(self, preds: torch.Tensor, labels: torch.Tensor):
        preds_np = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        acc = accuracy_score(labels_np, preds_np)
        f1 = f1_score(labels_np, preds_np)

        return acc, f1

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, preds, labels = self._shared_step(batch)

        acc, f1 = self._compute_metrics(preds, labels)

        self.log("train/loss", loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", acc, prog_bar=True, on_epoch=True)
        self.log("train/f1", f1, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, preds, labels = self._shared_step(batch)

        acc, f1 = self._compute_metrics(preds, labels)

        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/acc", acc, prog_bar=True, on_epoch=True)
        self.log("val/f1", f1, prog_bar=True, on_epoch=True)

        return {"val_loss": loss, "val_acc": acc, "val_f1": f1}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        return optimizer