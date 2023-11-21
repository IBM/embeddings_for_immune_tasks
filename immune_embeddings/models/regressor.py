import pytorch_lightning as pl
import torch
from torchmetrics.functional import mean_absolute_error, mean_squared_error
from tqdm.autonotebook import tqdm


class Regressor(pl.LightningModule):
    def __init__(self, model, *, lr=1.0e-3, lr_scheduler=None, lr_scheduler_kwargs=None, lr_scheduler_config=None,
                 label_aad=None, label_var=None):
        super().__init__()
        lr_scheduler = lr_scheduler or "ConstantLR"
        lr_scheduler_config = lr_scheduler_config or dict(interval="epoch", frequency=1)
        lr_scheduler_kwargs = lr_scheduler_kwargs or dict(factor=1.0)
        self.lr = lr
        self.lr_scheduler_config = lr_scheduler_config
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.lr_scheduler = lr_scheduler

        self.label_aad = label_aad
        self.label_var = label_var
        self.model = model

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler_cls = getattr(torch.optim.lr_scheduler, self.lr_scheduler)
        scheduler = lr_scheduler_cls(optim, **self.lr_scheduler_kwargs)
        return {"optimizer": optim, "lr_scheduler": {"scheduler": scheduler, **self.lr_scheduler_config}, }

    def compute_label_var_mean(self, train_dl):
        mean = 0.0
        var = 0.0
        n_points = len(train_dl.dataset)

        for batch in tqdm(train_dl, leave=False, desc="Variance calculation"):
            y = batch["y"]
            mean += y.sum() / n_points
            var += (y ** 2).sum() / (n_points - 1)
        var = var - n_points / (n_points - 1) * mean ** 2
        return var.cpu().item(), mean.cpu().item()

    def compute_label_aad(self, train_dl, mean: float):
        aad = 0.0
        n_points = len(train_dl.dataset)

        for batch in tqdm(train_dl, leave=False, desc="AAD calculation"):
            aad += (batch["y"] - mean).abs().sum() / n_points
        return aad.cpu().item()

    def setup_summary_stats(self):
        if self.trainer.train_dataloader is None:
            try:
                train_dl = self.trainer.datamodule.train_dataloader()
            except AttributeError:
                print("Train Dataloader not accessible. Skipping Summary Stats Calculation")
                return
        else:
            train_dl = self.trainer.train_dataloader
        if (self.label_var is None) or (self.label_aad is None):
            var, mean = self.compute_label_var_mean(train_dl)
            self.label_var = var
        if self.label_aad is None:
            aad = self.compute_label_aad(train_dl, mean=mean)
            self.label_aad = aad

    def on_fit_start(self) -> None:
        self.setup_summary_stats()

    def on_test_start(self) -> None:
        self.setup_summary_stats()

    def on_validation_start(self) -> None:
        self.setup_summary_stats()

    def step(self, batch, step):
        X = batch["X"]
        y = batch["y"]
        yhat = self.model(X).squeeze()
        y = y.squeeze()
        mse = mean_squared_error(yhat, y)
        mae = mean_absolute_error(yhat, y)
        loss = mse

        self.log(f"{step}/loss", loss, on_step=False, on_epoch=True)
        self.log(f"{step}/mae", mae, on_step=False, on_epoch=True)
        self.log(f"{step}/ae_reg_coeff", 1 - mae / self.label_aad, on_step=False, on_epoch=True)
        self.log(f"{step}/r2", 1 - mse / self.label_var, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def predict_step(self, batch, batch_idx):
        X = batch["X"]
        y = batch["y"]
        id_keys = [k for k in batch if k not in ["X", "y"]]
        return (torch.stack([self.model(X).squeeze(), y], -1), *[batch[k] for k in id_keys],)
