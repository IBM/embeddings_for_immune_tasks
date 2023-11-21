from abc import ABC, abstractmethod

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn


def setup_optimizer(model, optim_config):
    optimizer = getattr(torch.optim, optim_config["optimizer"])(
        model.parameters(), **optim_config["optimizer_params"]
    )
    return optimizer


def setup_metrics(metric_specs: dict):
    out_dict = dict()
    for k, v in metric_specs.items():
        if isinstance(v, str):
            out_dict[k] = getattr(torchmetrics, v)()
        else:
            cls_name = v['metric']
            cls_kwargs = v['metric_params']
            out_dict[k] = getattr(torchmetrics, cls_name)(**cls_kwargs)
    return nn.ModuleDict(out_dict)


class DictBasedPredictor:
    @staticmethod
    def concat_predict_dict(dl_results):
        ks = dl_results[0].keys()
        dl_results_concat = {k: [] for k in ks}
        for batch_dict in dl_results:
            for k in ks:
                dl_results_concat[k].append(batch_dict[k])
        for k in ks:
            if isinstance(dl_results_concat[k][0], torch.Tensor):
                try:
                    dl_results_concat[k] = torch.concat(dl_results_concat[k], dim=0)
                except RuntimeError:
                    pass
            else:
                try:
                    dl_results_concat[k] = np.concatenate(dl_results_concat[k], axis=0)
                except RuntimeError:
                    pass
        return dl_results_concat

    def on_predict_epoch_end(self, results):
        if isinstance(results[0], dict):
            return self.concat_predict_dict(results)
        else:
            for i, dl_results in enumerate(results):
                results[i] = self.concat_predict_dict(dl_results)
            return results


class BaseModel(pl.LightningModule, ABC):
    def __init__(self, *, optimization: dict = None, metrics: dict = None, loss: nn.Module):
        super(BaseModel, self).__init__()
        self.metrics = setup_metrics(metrics or dict())
        self.loss = loss

        self.optimization = optimization or {"optimizer": "Adam", "optimizer_params": {"lr": 1.0e-3}}

    def log_step_metrics(self, preds, target, phase):
        with torch.no_grad():
            for metric_name, metric_obj in self.metrics.items():
                if not isinstance(metric_obj, torchmetrics.Metric):
                    self.log(f"{phase}/{metric_name}", metric_obj(preds=preds, target=target))
                else:
                    metric_obj.update(preds=preds, target=target)

    def log_epoch_metrics(self, phase):
        with torch.no_grad():
            for metric_name, metric_obj in self.metrics.items():
                if isinstance(metric_obj, torchmetrics.Metric):
                    self.log(f"{phase}/{metric_name}", metric_obj.compute())

    def compute_loss(self, *loss_args, phase):
        loss = self.loss(*loss_args)
        self.log(f"{phase}/loss", loss)
        return loss

    def configure_optimizers(self):
        return setup_optimizer(self, self.optimization)

    @abstractmethod
    def step(self, batch, phase):
        pass

    def on_train_epoch_end(self) -> None:
        return self.log_epoch_metrics("train")

    def on_test_epoch_end(self) -> None:
        return self.log_epoch_metrics("test")

    def on_validation_epoch_end(self) -> None:
        return self.log_epoch_metrics("val")

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, "test")
