from typing import Any, Dict, Tuple

import torch
import dgl
from torch.nn import functional as F

from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from components import SRCLML


class SRCLMLLitModule(LightningModule):
    def __init__( self, net: torch.nn.Module, lambda1, lambda2) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net

        self.criterion = self.net.loss_func

        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()

    def forward(self, x: dgl.DGLGraph, t: torch.Tensor) -> torch.Tensor:
        return self.net(x, t)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def bpr_loss(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))

    def sample_pairs(self, interaction_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_indices = interaction_matrix.nonzero(as_tuple=False)
        neg_indices = (interaction_matrix == 0).nonzero(as_tuple=False)
        pos_samples = pos_indices[torch.randint(len(pos_indices), (len(pos_indices),))]
        neg_samples = neg_indices[torch.randint(len(neg_indices), (len(pos_indices),))]
        return pos_samples, neg_samples

    def model_step(
        self, batch: Tuple[Dict[str, Any], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, y = batch
        text_features = inputs['text']
        graph_features = inputs['graph']

        logits = self.forward(text_features, graph_features)
        
        O_as, O_at, O_st = inputs['O_as'], inputs['O_at'], inputs['O_st']
        pos_as, neg_as = self.sample_pairs(O_as)
        pos_at, neg_at = self.sample_pairs(O_at)
        pos_st, neg_st = self.sample_pairs(O_st)

        pos_scores_as = logits[pos_as[:, 0], pos_as[:, 1]]
        neg_scores_as = logits[neg_as[:, 0], neg_as[:, 1]]
        pos_scores_at = logits[pos_at[:, 0], pos_at[:, 1]]
        neg_scores_at = logits[neg_at[:, 0], neg_at[:, 1]]
        pos_scores_st = logits[pos_st[:, 0], pos_st[:, 1]]
        neg_scores_st = logits[neg_st[:, 0], neg_st[:, 1]]

        loss_as = self.bpr_loss(pos_scores_as, neg_scores_as)
        loss_at = self.bpr_loss(pos_scores_at, neg_scores_at)
        loss_st = self.bpr_loss(pos_scores_st, neg_scores_st)

        loss = loss_as + loss_at + loss_st \
            + self.lambda1 * (loss_a + loss_s)
        
        preds = torch.argmax(logits, dim=1)
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
