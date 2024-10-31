from typing import Any, Dict, Tuple

import torch
import dgl
from torch.nn import functional as F

from lightning import LightningModule
from torch_geometric.data import HeteroData
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from components.SRCLML import SRCLML

class SRCLMLLitModule(LightningModule):
    def __init__(
            self,
            net: torch.nn.Module,
            invoke_subgraph, app_tag_subgraph, api_tag_subgraph,
            embd_dim, num_layers,
        ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = SRCLML(
            invoke_subgraph, app_tag_subgraph, api_tag_subgraph,
            embd_dim, num_layers,
        )

        self.criterion = self.net.loss_func

        # torchmetrics
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()

    def forward(self, x, t: torch.Tensor) -> torch.Tensor:
        invoke_sample, app_tag_sample, api_tag_sample = x
        return self.net(invoke_sample, app_tag_sample, api_tag_sample)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[Dict[str, Any], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # SRCLML 无需采样
        inputs, y = batch
        text_features = inputs['text']
        graph_features = inputs['graph']

        logits = self.forward(text_features, graph_features)
        loss = torch.tensor(1, ) # TODO

        preds = torch.argmax(logits, dim=1)
        # TODO: paper 3.6 -> preds
        return loss, preds, y

    def training_step(
        self, batch: Tuple[Dict[str, Any], torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[Dict[str, Any], torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[Dict[str, Any], torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        return {"optimizer": optimizer}

if __name__ == "__main__":
    pass
