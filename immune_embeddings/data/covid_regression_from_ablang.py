from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from random import randint

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split

from immune_embeddings.data import DataDir


def get_covid_ablang_datamodule(label_jiggle=0.05, subsample_frac=None, batch_size=128, num_workers=0,
                                seed=None, dl_kwargs=None):
    if dl_kwargs is None:
        dl_kwargs = dict()

    return CovidDataModule(CovidDataDir(seed=seed, subsample_frac=subsample_frac),
                           label_jiggle=label_jiggle, batch_size=batch_size, num_workers=num_workers,
                           dl_kwargs=dl_kwargs, seed=seed)


class CovidDataModule(pl.LightningDataModule):
    def __init__(self, datadir, label_jiggle=0.05, batch_size=128, num_workers=0, dl_kwargs=None,
                 label_col="target_value", embedding_col="hl_concat_ablang_embedding", seed=None):
        super(CovidDataModule, self).__init__()
        if seed is None:
            seed = randint(10000, 100000)
        self.seed = seed
        self.label_jiggle = label_jiggle
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datadir = datadir
        self.label_col = label_col
        self.embedding_col = embedding_col
        if dl_kwargs is None:
            dl_kwargs = dict()
        self.dl_kwargs = dl_kwargs

    def train_dataloader(self):
        return self.datadir.get_dataloader(
            split="train", label_col=self.label_col, embedding_col=self.embedding_col, label_jiggle=self.label_jiggle,
            batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, seed=self.seed, **self.dl_kwargs)

    def val_dataloader(self):
        return self.datadir.get_dataloader(
            split="val", label_col=self.label_col, embedding_col=self.embedding_col, label_jiggle=self.label_jiggle,
            batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, seed=self.seed, **self.dl_kwargs)

    def test_dataloader(self):
        return self.datadir.get_dataloader(
            split="test", label_col=self.label_col, embedding_col=self.embedding_col, label_jiggle=0.0,
            batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, seed=self.seed, **self.dl_kwargs)


@dataclass
class CovidDataDir(DataDir):
    data_relpath: Path = Path("antibody") / "covid_alphaseq"
    seed: int = field(default_factory=partial(randint, 10000, 100000))
    subsample_frac: float = None
    _raw_df: pd.DataFrame = None
    _seq_split: pd.DataFrame = None
    _data_df: pd.DataFrame = None

    @property
    def raw_df(self):
        if self._raw_df is None:
            self._raw_df = pd.read_pickle(self.data_dir / "ablang_embeddings_hc_and_lc_df.pkl")
            if self.subsample_frac is not None:
                self._raw_df = self.raw_df.sample(frac=self.subsample_frac, random_state=self.seed)
        return self._raw_df

    @property
    def seq_split(self):
        if self._seq_split is None:
            self._seq_split = self.get_seq_split()
        return self._seq_split

    def get_seq_split(self):
        seq_aff = self.clean_df.groupby("Sequence").Pred_affinity.mean().reset_index()
        seq_aff = seq_aff.assign(pa_q=pd.qcut(seq_aff.Pred_affinity, 20, labels=False))

        seq_aff = seq_aff.assign(pa_q=pd.qcut(seq_aff.Pred_affinity, 20, labels=False))
        seq_train, seq_test = train_test_split(seq_aff, stratify=seq_aff.pa_q, random_state=self.seed)
        seq_train, seq_val = train_test_split(seq_train, stratify=seq_train.pa_q, random_state=self.seed)

        seq_train = seq_train.assign(split="train")
        seq_val = seq_val.assign(split="val")
        seq_test = seq_test.assign(split="test")
        seq_split = pd.concat([seq_train, seq_test, seq_val])
        return seq_split

    @property
    def clean_df(self):
        df_data = self.raw_df.loc[~self.raw_df.Pred_affinity.isna()]
        return df_data.loc[df_data.Target == "MIT_Target"]

    @property
    def data_df(self):
        if self._data_df is None:
            self._data_df = self.get_data_df()
        return self._data_df

    def get_data_df(self):
        df_split = self.clean_df.merge(self.seq_split[["Sequence", "split"]], on="Sequence")

        self.check_data_split(df_split)

        df_split = df_split.assign(hl_concat_ablang_embedding=df_split.apply(
            lambda row: np.concatenate([row.lc_ablang_embedding, row.hc_ablang_embedding]), axis=1))
        # Normalizing label distribution:
        pa_train = df_split.loc[df_split.split == "train"].Pred_affinity
        mean = np.mean(pa_train)
        std = np.mean(pa_train)
        data_df = df_split.assign(target_value=(df_split.Pred_affinity - mean) / std)
        return data_df

    def check_data_split(self, df: pd.DataFrame):
        train_seqs = df.loc[df.split == "train"].Sequence.unique()
        test_seqs = df.loc[df.split == "test"].Sequence.unique()
        val_seqs = df.loc[df.split == "val"].Sequence.unique()

        train_seqs = set(train_seqs)
        test_seqs = set(test_seqs)
        val_seqs = set(val_seqs)

        assert len(train_seqs.intersection(test_seqs)) == 0
        assert len(train_seqs.intersection(val_seqs)) == 0
        assert len(val_seqs.intersection(test_seqs)) == 0

    def get_dataset(self, split="train", label_jiggle=0.05, label_col="target_value",
                    embedding_col="hl_concat_ablang_embedding"):
        return EmbeddingDataset(self.data_df, split=split, label_jiggle=label_jiggle, label_col=label_col,
                                embedding_col=embedding_col)

    def get_dataloader(self, split="train", label_col="target_value", embedding_col="hl_concat_ablang_embedding",
                       label_jiggle=0.05, batch_size=128, num_workers=0, shuffle=None, seed=None, **dl_kwargs):
        if shuffle is None:
            shuffle = split == "train"
        if seed is None:
            seed = randint(10000, 100000)

        generator = torch.Generator()
        generator.manual_seed(seed)

        return torch.utils.data.DataLoader(
            self.get_dataset(split=split, label_jiggle=label_jiggle, label_col=label_col, embedding_col=embedding_col),
            batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, generator=generator, **dl_kwargs)


def encode_uuid(hexstr):
    return np.array(np.frombuffer(bytes.fromhex(hexstr), dtype="int64"))


def decode_uuid(uuid_array):
    return uuid_array.tobytes().hex()


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, split="train", label_col="target_value", embedding_col="hl_concat_ablang_embedding",
                 hex_ids=("seq_uuid", "lc_uuid", "hc_uuid"), label_jiggle=0.05):
        super().__init__()
        dataframe = dataframe.loc[dataframe.split == split]

        assert np.all(dataframe.split == split)

        self.embed = np.stack(dataframe[embedding_col].values, 0)
        self.label = dataframe[label_col].values
        self.ids = {hexcol: dataframe[hexcol].values for hexcol in hex_ids}
        self.splits = dataframe.split.values
        self.label_jiggle = label_jiggle

    def __len__(self):
        return len(self.embed)

    def __getitem__(self, i):
        y = self.label[i]
        if self.label_jiggle:
            y = y + np.random.randn() * self.label_jiggle
        return {
            "X": self.embed[i].astype(np.float32), "y": y.astype(np.float32),
            **{hexcol: encode_uuid(self.ids[hexcol][i]) for hexcol in self.ids}, }
