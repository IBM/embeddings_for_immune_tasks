from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from immune_embeddings.data import DataDir
from immune_embeddings.data.covid_to_esm import CovidSequenceDataset, PredictDataModule


@dataclass
class GuikemaBCellDataDir(DataDir):
    data_relpath: Path = Path("bcr") / "bcells_guikema" / "sequences"
    seq_col: str = "Sequence_AA"
    id_col: str = "Sequence ID"

    def __getitem__(self, item):
        datafile_path = sorted(list(self.data_dir.glob("*Nt*.csv")))[item]
        df = pd.read_csv(datafile_path, sep="\t")
        df = df.loc[~df[self.seq_col].str.contains("\*")]
        return df


def get_bcr_guikema_for_esm(batch_size, *, sequence_column="Sequence_AA", id_column="Sequence ID",
                            model_id="esm2_t33_650M_UR50D", file_number=0, total_splits=5, current_split=0,
                            truncation_seq_length=None, num_workers=0, shuffle=False, **dataloader_extra_kws):
    data_dir = GuikemaBCellDataDir(seq_col=sequence_column, id_col=id_column)
    df = data_dir[file_number]
    split_df = np.array_split(df, total_splits)[current_split]

    dataset = CovidSequenceDataset(df=split_df, seq_column=data_dir.seq_col, id_column=data_dir.id_col,
                                   id_transform=None)

    return PredictDataModule(
        DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                   **dataloader_extra_kws),
        esm_tokenizer_params=dict(model_id=model_id, truncation_seq_length=truncation_seq_length))
