# Data Description

## Raw dataset
- Filename: `MITLL_AAlphaBio_Ab_Binding_dataset.csv`
- Origin: `https://zenodo.org/records/5095284` (CC-BY)

# Sequence IDS
- Filename: `sequence_uuids.csv`
- columns `Sequence,seq_uuid`
- Origin: Create a unique UUID for each unique sequence in `MITLL_AAlphaBio_Ab_Binding_dataset.csv`


- Filename: `hc_uuids.csv`
- columns `HC,hc_uuid`
- Origin: Create a unique UUID for each unique heavy chain (HC) in `MITLL_AAlphaBio_Ab_Binding_dataset.csv`


- Filename: `lc_uuids.csv`
- columns `LC,hc_uuid`
- Origin: Create a unique UUID for each unique heavy chain (LC) in `MITLL_AAlphaBio_Ab_Binding_dataset.csv`

## Ablang Embeddings

- Filename `ablang_embeddings_hc_and_lc_df.pkl`
- Generation: `bench/alphaseq_covid_embeddings.ipynb`

## ESM Embeddings

- Filenames: `"esm_covid_[model_size]_layer[L]_hc_and_lc.pkl"`
- Generation: CLI with configs like `experiments/covid_esm/hc_650M/covid_esm_650M_hc_0.json`
