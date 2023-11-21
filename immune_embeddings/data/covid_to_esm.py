import esm
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

from immune_embeddings.data import covid_regression_from_ablang as covid_data_tools


class PredictDataModule(pl.LightningDataModule):
    def __init__(self, *dataloaders: DataLoader, esm_tokenizer_params=None):
        esm_tokenizer_params = esm_tokenizer_params or dict()

        self.tokenizer = EmbedTokensESM(**esm_tokenizer_params)

        super(PredictDataModule, self).__init__()
        self.dataloaders = dataloaders

    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        token_dict = self.tokenizer(batch["protein_sequences"], batch["sequence_ids"])
        return dict(**batch, **token_dict)

    def predict_dataloader(self):
        return self.dataloaders


class EmbedTokensESM:
    def __init__(self, model_id="esm2_t33_650M_UR50D", truncation_seq_length=None):
        assert model_id.startswith("esm2"), "Only ESM2 is supported"
        self.model_id = model_id
        self.truncation_seq_length = truncation_seq_length
        self.alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        self.batch_converter = self.alphabet.get_batch_converter(truncation_seq_length=truncation_seq_length)

    def __call__(self, protein_sequences, sequence_ids):
        raw_batch = list(zip(sequence_ids, protein_sequences))
        batch_labels, batch_strs, batch_tokens = self.batch_converter(raw_batch)
        protein_lens = torch.tensor([len(self.alphabet.tokenize(p)) for p in protein_sequences])
        return dict(protein_tokens=batch_tokens, protein_lens=protein_lens)


def identity_transform(x):
    return x


class CovidSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_column="Sequence", id_column="seq_uuid",
                 id_transform=covid_data_tools.encode_uuid):
        data = df[[seq_column, id_column]].drop_duplicates()
        self.sequences = data[seq_column].values

        if id_transform is None:
            id_transform = identity_transform

        self.ids = data[id_column].apply(id_transform).values

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        return {"protein_sequences": self.sequences[item], "sequence_ids": self.ids[item], }


def get_covid_for_esm(
        batch_size, *, sequence_column="Sequence", id_column="seq_uuid", model_id="esm2_t33_650M_UR50D", total_splits=5,
        current_split=0, truncation_seq_length=None, num_workers=0, shuffle=False, **dataloader_extra_kws):
    data_dir = covid_data_tools.CovidDataDir()
    data_df = data_dir.data_df
    data_df = np.array_split(data_df, total_splits)[current_split]

    dataset = CovidSequenceDataset(df=data_df, seq_column=sequence_column, id_column=id_column)

    return PredictDataModule(
        DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                   **dataloader_extra_kws),
        esm_tokenizer_params=dict(model_id=model_id, truncation_seq_length=truncation_seq_length))
