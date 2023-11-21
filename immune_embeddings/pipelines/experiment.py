from copy import deepcopy
from pathlib import Path
import os

import dotenv
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger

from immune_embeddings.pipelines.data_registry import data_registry
from immune_embeddings.pipelines.model_registry import model_registry
from immune_embeddings.utils import flatten_to_string, save_config


def run_experiment(config):
    raw_config = deepcopy(config)

    experiment_root = Path(config.get("experiment_root", Path.cwd()))
    experiment_name = config.get("experiment_name", "immune_embedding_experiment")
    trainer_params = config.get("trainer_params", dict())
    do_train = "train_model" in config and config["train_model"]
    do_predict = "predict_model" in config and config["predict_model"]
    do_test = "test_model" in config and config["test_model"]
    do_validate = "validate_model" in config and config["validate_model"]

    assert do_train or do_predict or do_test or do_validate, (
        "At least one True: [train_model, predict_model, test_model, validate_model]")
    dotenv.load_dotenv(config['env_fpath'], override=True)
    if "mlflow_artifacts_uri" in config:
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = config["mlflow_artifacts_uri"]
    mlflogger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=config.get("mlflow_uri", None))
    run_id = mlflogger.run_id
    config["mlflow_run_id"] = run_id

    experiment_path = experiment_root / experiment_name / str(run_id)
    experiment_path.mkdir(exist_ok=False, parents=True)

    pl.seed_everything(config.get('global_seed', 123))

    trainer_params.update(default_root_dir=str(experiment_path))
    if do_train:
        checkpoint_params = config.get("checkpoint_params",
                                       {"monitor": "valid/loss", "every_n_epochs": 10, "save_top_k": 3})
        checkpoint_params.update(dirpath=str(experiment_path), filename="best")
        early_stopping = config.get("early_stopping", False)
        config.update(early_stopping=early_stopping)
        if early_stopping:
            early_stopping_params = config.get("early_stopping_params", dict(monitor=checkpoint_params["monitor"]))
            config.update(early_stopping_params=early_stopping_params)

    mlflogger.log_hyperparams(params=flatten_to_string(config))
    processed_config_no_model = deepcopy(config)

    # Starting to add objects to config, loss of serializability
    trainer_params.update(logger=mlflogger)

    model_params = config["models"]
    data_params = config["data"]
    model = model_registry.get(model_params)
    data_module: pl.LightningDataModule = data_registry.get(data_params)

    callbacks = []
    if do_train:
        checkpoint = pl.callbacks.ModelCheckpoint(**checkpoint_params)
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
        callbacks.extend([checkpoint, lr_monitor])
        if early_stopping:
            early_stopping_callback = pl.callbacks.EarlyStopping(**early_stopping_params)
            callbacks.append(early_stopping_callback)

    trainer_params.update(callbacks=callbacks)
    trainer = pl.Trainer(**trainer_params)

    # Logging setup info
    mlflogger.experiment.log_dict(run_id, raw_config, "raw_config.json")
    mlflogger.experiment.log_dict(run_id, processed_config_no_model, "processed_config.json")
    save_config(processed_config_no_model, experiment_path / "processed_config.json")
    save_config(raw_config, experiment_path / "raw_config.json")

    if do_train:
        trainer.fit(model, datamodule=data_module)
        mlflogger.experiment.log_artifact(run_id, str(experiment_path / "best.ckpt"))

    if do_validate:
        trainer.validate(model, datamodule=data_module)

    if do_test:
        if do_train:
            ckpt_path = "best"
        else:
            ckpt_path = None
        trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)

    if do_predict:
        results = trainer.predict(model, datamodule=data_module)

        predfile = str(experiment_path / "predictions.pt")
        torch.save(results, predfile)
        mlflogger.experiment.log_artifact(run_id, predfile)
