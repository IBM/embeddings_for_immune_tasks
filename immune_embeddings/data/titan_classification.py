from abc import ABC
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from random import randint

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from immune_embeddings.data import DataDir
from immune_embeddings.data.data_utils import SplitDataModule


class RebalancingDataset(Dataset, ABC):
    def __init__(self, X: np.ndarray, y: np.ndarray, seed: int = None):
        self.X_raw = X
        self.y_raw = y
        assert len(y.shape) == 1
        assert len(X.shape) > 1
        assert y.shape[0] == X.shape[0]

        self.labels, self.label_counts = np.unique(y, return_counts=True)
        self.n_labels = len(self.labels)
        self.rng = np.random.default_rng(seed)


class BootstrappedBalancedDataset(RebalancingDataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seed: int = None, n_bootstrap_per_class=None,
                 bootstrap_factor=None, mode=None):
        super(BootstrappedBalancedDataset, self).__init__(X, y, seed=seed)
        if n_bootstrap_per_class is None:
            if bootstrap_factor is None:
                bootstrap_factor = 1
            if mode is None:
                mode = "expand"
            if mode == "expand":
                n_bootstrap_per_class = int(max(self.label_counts) * bootstrap_factor)
            elif mode == "shrink":
                assert bootstrap_factor <= 1
                n_bootstrap_per_class = int(min(self.label_counts) * bootstrap_factor)
            else:
                raise ValueError("mode can be 'expand' or 'shrink'")
        else:
            assert (bootstrap_factor is None), "bootstrap_factor incompatible with n_bootstrap_per_class"
            assert mode is None, "mode incompatible with n_bootstrap_per_class"

        self.n_bootstrap_per_class = n_bootstrap_per_class
        self.sampled_indices = None
        self.resample_indices()

    def resample_indices(self):
        sampled_indices = []
        for label in self.labels:
            label_idxs = np.argwhere(self.y_raw == label).squeeze()
            sampled_indices.append(self.rng.choice(label_idxs, self.n_bootstrap_per_class, replace=True))
        self.sampled_indices = np.concatenate(sampled_indices)

    def __getitem__(self, item):
        return {"X": self.X_raw[self.sampled_indices[item]], "y": self.y_raw[self.sampled_indices[item]], }

    def __len__(self):
        return len(self.sampled_indices)


class SubSamplingClassificationDataset(RebalancingDataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, rechunk_every_n=None, seed=None):
        super(SubSamplingClassificationDataset, self).__init__(X, y, seed=seed)
        self.access_counter = 0
        self.rechunk_counter = 0
        if rechunk_every_n is None:
            rechunk_every_n = len(X)
        self.rechunk_every_n = rechunk_every_n

        self.n_per_label = min(self.label_counts)
        self.len_ = self.n_per_label * self.n_labels

        self.label_indices = []
        for li in self.labels:
            self.label_indices.append(np.argwhere(y == li).squeeze())
        self.label_index_chunks = []
        self.shuffle_chunks()

    def shuffle_chunks(self):
        self.label_index_chunks = [np.array_split(self.rng.permutation(label_index_list), self.n_per_label) for
                                   label_index_list in self.label_indices]

    def get_random_item_idx(self, label_id, item_id):
        return self.rng.choice(self.label_index_chunks[label_id][item_id])

    def __len__(self):
        return self.len_

    def __getitem__(self, item):
        assert isinstance(item, int), f"{self.__class__} supports only integer indexing (no slices!)"
        assert item < len(self)

        if self.access_counter >= self.rechunk_every_n:
            self.access_counter = 0
            self.rechunk_counter += 1
            self.shuffle_chunks()

        label_id = item % self.n_labels
        item_id = item // self.n_labels
        raw_idx = self.get_random_item_idx(label_id, item_id)
        label = self.labels[label_id]
        y = self.y_raw[raw_idx]
        assert label == y, "Something is wrong with indexing"

        self.access_counter += 1

        return {"X": self.X_raw[raw_idx], "y": y}


@dataclass
class TitanDataDir(DataDir):
    data_relpath: Path = Path("tcr") / "raw_cdr"
    seed: int = field(default_factory=partial(randint, 10000, 100000))
    test_size: float = 0.2
    val_size: float = 0.2
    seq_column: str = "TRB_CDR3"
    embedding_column: str = "TRB_CDR3_TCRBERT"
    label_column: str = "Label"
    data_filename: str = "dresden_tcrbert.pkl"
    standardize_embeddings: bool = True

    _seq_split: pd.DataFrame = None
    _data_df: pd.DataFrame = None
    _clean_df: pd.DataFrame = None

    def check_filename(self):
        assert (self.data_dir / self.data_filename).exists()

    @property
    def df_with_splits(self):
        if self._seq_split is None:
            self._seq_split = self.get_splits()
        return self._seq_split

    def get_splits(self):
        train_df, test_df = train_test_split(self.clean_df, test_size=self.test_size, stratify=self.clean_df["Label"],
                                             shuffle=True, random_state=self.seed)
        train_df, val_df = train_test_split(train_df, test_size=self.val_size, stratify=train_df["Label"], shuffle=True,
                                            random_state=self.seed)
        train_df = train_df.assign(split="train")
        test_df = test_df.assign(split="test")
        val_df = val_df.assign(split="val")
        df_with_splits = pd.concat([train_df, test_df, val_df], ignore_index=True)
        return df_with_splits

    @property
    def clean_df(self):
        def safe_binary_label(row):
            if row == 1:
                return 1
            elif row == -1:
                return 0
            else:
                raise ValueError("Labels +/-1 expected")

        if self._clean_df is None:
            self.check_filename()
            self._clean_df = pd.read_pickle(self.data_dir / self.data_filename)
            self._clean_df = self._clean_df.assign(
                **{self.label_column: self._clean_df[self.label_column].apply(safe_binary_label)})
        return self._clean_df

    @property
    def data_df(self):
        if self._data_df is None:
            self._data_df = self.get_data_df()
        return self._data_df

    def preprocess_embeddings(self, df_with_splits: pd.DataFrame):
        if not self.standardize_embeddings:
            return df_with_splits
        train_mask = df_with_splits.split == "train"
        train_df = df_with_splits.loc[train_mask]
        other_df = df_with_splits.loc[~train_mask]
        train_embeddings = np.stack(train_df[self.embedding_column].values, 0)
        mean = train_embeddings.mean(axis=0, keepdims=True)
        std = train_embeddings.std(axis=0, keepdims=True)
        other_embeddings = np.stack(other_df[self.embedding_column].values, 0)
        train_embeddings = (train_embeddings - mean) / std
        other_embeddings = (other_embeddings - mean) / std

        train_df = train_df.assign(**{self.embedding_column: list(train_embeddings)})
        other_df = other_df.assign(**{self.embedding_column: list(other_embeddings)})
        return pd.concat([train_df, other_df])

    def get_data_df(self):
        data_with_splits = self.df_with_splits
        processed_data = self.preprocess_embeddings(data_with_splits)
        return processed_data[
            [self.seq_column, self.embedding_column, self.label_column, "split"]
        ]

    def get_train_dataset(self, rechunk_every_n=None):
        train_df = self.data_df.loc[self.data_df.split == "train"]
        return SubSamplingClassificationDataset(
            X=np.stack(train_df[self.embedding_column].values, axis=0), y=train_df[self.label_column].values,
            seed=self.seed, rechunk_every_n=rechunk_every_n)

    def get_val_dataset(self, bootstrap_factor=1, mode="expand"):
        val_df = self.data_df.loc[self.data_df.split == "val"]
        return BootstrappedBalancedDataset(
            X=np.stack(val_df[self.embedding_column].values, axis=0), y=val_df[self.label_column].values,
            seed=self.seed, n_bootstrap_per_class=None, bootstrap_factor=bootstrap_factor, mode=mode)

    def get_test_dataset(self, bootstrap_factor=1, mode="expand"):
        test_df = self.data_df.loc[self.data_df.split == "test"]
        return BootstrappedBalancedDataset(
            X=np.stack(test_df[self.embedding_column].values, axis=0), y=test_df[self.label_column].values,
            seed=self.seed, n_bootstrap_per_class=None, bootstrap_factor=bootstrap_factor, mode=mode)

    def get_dataset(self, split="train", **ds_kwargs):
        if split == "train":
            return self.get_train_dataset(**ds_kwargs)
        elif split == "val":
            return self.get_val_dataset(**ds_kwargs)
        elif split == "test":
            return self.get_test_dataset(**ds_kwargs)
        else:
            raise ValueError(f"Incorrect argument split={split}. Accepting 'train', 'val' or 'test'.")

    def get_dataloader(self, split="train", batch_size=128, num_workers=0, shuffle=None, seed=None, ds_kwargs=None,
                       **dl_kwargs):
        if shuffle is None:
            shuffle = split == "train"
        if seed is None:
            seed = randint(10000, 100000)

        generator = torch.Generator()
        generator.manual_seed(seed)

        if ds_kwargs is None:
            ds_kwargs = dict()

        return torch.utils.data.DataLoader(self.get_dataset(split=split, **ds_kwargs), batch_size=batch_size,
                                           num_workers=num_workers, shuffle=shuffle, generator=generator, **dl_kwargs)


def get_titan_embedding_classification_datamodule(
        test_size: float = 0.2, val_size: float = 0.2, chain: str = "B", model: str = "tcrbert", esm_size: str = "650M",
        layer: int = None, batch_size=128, standardize_embeddings: bool = True, num_workers=0, seed=None,
        train_ds_kwargs=None, test_ds_kwargs=None, val_ds_kwargs=None):
    if train_ds_kwargs is None:
        train_ds_kwargs = dict()
    if test_ds_kwargs is None:
        test_ds_kwargs = dict()
    if val_ds_kwargs is None:
        val_ds_kwargs = dict()

    if seed is None:
        seed = randint(10000, 100000)

    if model == "tcrbert":
        config = get_tcrbert_config(chain, repr_layer=layer)
    elif model == "esm":
        config = get_esm_config(chain=chain, model_size=esm_size, repr_layer=layer)
    else:
        raise ValueError("model must be one of 'tcrbert', 'esm'")

    datadir = TitanDataDir(seed=seed, test_size=test_size, val_size=val_size,
                           standardize_embeddings=standardize_embeddings, **config)

    train_ds = datadir.get_dataset("train", **train_ds_kwargs)
    val_ds = datadir.get_dataset("val", **val_ds_kwargs)
    test_ds = datadir.get_dataset("test", **test_ds_kwargs)

    dl = SplitDataModule(train_set=train_ds, valid_set=val_ds, test_set=test_ds, num_workers=num_workers,
                         batch_size=batch_size, random_seed=seed)
    return dl


def get_tcrbert_config(chain="B", repr_layer=None):
    assert chain in ["A", "B"]
    assert repr_layer is None or isinstance(repr_layer, int)
    layer_suffix = f"_L{repr_layer}" if repr_layer else ""

    return dict(seq_column=f"TR{chain}_CDR3", embedding_column=f"TR{chain}_CDR3_TCRBERT" + layer_suffix.upper(),
                label_column="Label", data_filename=f"dresden_tcrbert{layer_suffix.lower()}.pkl")


def get_esm_config(chain="B", model_size="650M", repr_layer=None):
    default_layer_map = {"3B": 36, "650M": 33, "35M": 12}
    if repr_layer is None:
        repr_layer = default_layer_map[model_size]
    assert chain in ["A", "B"]
    return dict(seq_column=f"TR{chain}_CDR3", embedding_column=f"TR{chain}_CDR3_ESM", label_column="Label",
                data_filename=f"dresden_esm_{model_size}_L{repr_layer}.pkl")


def get_dimensions(model: str, esm_size=None):
    if model == "tcrbert":
        return 768
    elif model == "esm":
        esm_size_map = {"35M": 480, "650M": 1280, "3B": 2560}
        return esm_size_map[esm_size]
    raise ValueError("Model must be 'tcrbert' or 'esm'")
