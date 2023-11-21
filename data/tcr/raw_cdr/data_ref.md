# Data Description

## Raw data
- Filename: `dresden_tcr.csv`
- Origin: `https://github.com/viragbioinfo/IMMREP_2022_TCRSpecificity`
- Procedure: combine the training and test data for the epitope `GILGFVFTL`
- Required columns: `Label, TRB_CDR3, TRA_CDR3`

## TCR-BERT Embeddings
- Filenames: `dresden_tcrbert_LX.pkl` with X the layer
- Origin: Created with `bench/bench/tcr_dresden_embed_tcrbert.ipynb`

## ESM2 Embeddings
- Filenames: `dresden_esm_XXXX_LY.pkl` with XXXX the model size (e.g. '650M') and Y the layer
- Origin: Created with `bench/tcr_dresden_embed_esm.ipynb`
