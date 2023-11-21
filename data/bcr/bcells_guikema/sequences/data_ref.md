# Data Description

## B-Cell Data

- Filenames: `sampleXX_Nt_info.csv` (`XX` is a number)
- Origin: `https://vdjserver.org/community/8899006209436478995-242ac118-0001-012`
- Processing: each `GYY_Z.tsv` file from the source is filtered for `Functionality == 'productive'` and the nucleotide
  sequence column `Sequence` is converted to an amino-acid sequence. The resulting table is saved
  as `sampleXX_Nt_info.csv`

## ESM Embeddings

Created from the `experiments/guikema_esm/guikema_esm_base.json` pipeline config

## Ablang Embeddings

Created with `bench/bcr_ablang_embeddings.ipynb`
