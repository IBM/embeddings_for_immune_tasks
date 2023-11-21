from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from immune_embeddings.data.covid_regression_from_ablang import CovidDataDir as AblangCovidDataDir
from immune_embeddings.data.covid_regression_from_ablang import (EmbeddingDataset, CovidDataModule)


@dataclass
class CovidDataDir(AblangCovidDataDir):
    data_relpath: Path = Path("antibody") / "covid_alphaseq"
    model_size: str = "650M"
    layer_repr: int = 6
    _clean_df: pd.DataFrame = None

    @property
    def data_filename(self):
        return f"esm_covid_{self.model_size}_layer{self.layer_repr}_hc_and_lc.pkl"

    def check_filename(self):
        assert (self.data_dir / self.data_filename).exists()

    @property
    def clean_df(self):
        if self._clean_df is None:
            self.check_filename()
            self._clean_df = pd.read_pickle(self.data_dir / self.data_filename)
            if self.subsample_frac is not None:
                self._clean_df = self._clean_df.sample(frac=self.subsample_frac, random_state=self.seed)
        return self._clean_df

    def get_data_df(self):
        df_split = self.clean_df.merge(self.seq_split[["Sequence", "split"]], on="Sequence",
                                       suffixes=("_fromfile", ""))

        self.check_data_split(df_split)

        # Normalizing label distribution:
        pa_train = df_split.loc[df_split.split == "train"].Pred_affinity
        mean = np.mean(pa_train)
        std = np.mean(pa_train)
        data_df = df_split.assign(target_value=(df_split.Pred_affinity - mean) / std)
        return data_df

    def get_dataset(self, split="train", label_jiggle=0.05, label_col="target_value",
                    embedding_col="hc_lc_concat_esm"):
        return EmbeddingDataset(self.data_df, split=split, label_jiggle=label_jiggle, label_col=label_col,
                                embedding_col=embedding_col, hex_ids=("seq_uuid",))

    def get_dataloader(self, split="train", label_col="target_value", embedding_col="hc_lc_concat_esm",
                       label_jiggle=0.05, batch_size=128, num_workers=0, shuffle=None, seed=None, **dl_kwargs):
        return super(CovidDataDir, self).get_dataloader(split, label_col, embedding_col, label_jiggle, batch_size,
                                                        num_workers, shuffle, seed, **dl_kwargs)


def get_covid_esm_datamodule(model_size="650M", layer_repr=6, label_jiggle=0.05, subsample_frac=None,
                             batch_size=128, num_workers=0, seed=None, dl_kwargs=None):
    if dl_kwargs is None:
        dl_kwargs = dict()

    return CovidDataModule(
        CovidDataDir(model_size=model_size, layer_repr=layer_repr, subsample_frac=subsample_frac, seed=seed),
        label_jiggle=label_jiggle, batch_size=batch_size, num_workers=num_workers, dl_kwargs=dl_kwargs, seed=seed,
        embedding_col="hc_lc_concat_esm")
