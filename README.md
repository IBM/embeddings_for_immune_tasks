# Immune Protein Embeddings

This is the code used
for [Do Domain-Specific Protein Language Models Outperform General Models on 
Immunology-Related Tasks?](https://www.biorxiv.org/content/10.1101/2023.10.17.562795v3)

##  Installation

We recommend using a virtual environment in which to install the python package.

```shell
python -m venv immune_embbedings_venv
source ./immune_embeddings_venv/bin/activate

pip install -e .
```

## Usage

Model training and inference is obtained by using the `immune_embeddings.pipelines` command-line interface,
which takes a JSON configuration file as input, specifying the dataset, task and model architecture.
Examples for each task are provided in the `experiments` directory. Note that these files contain "INSERT VALUE" values
which must be provided by the user. In particular, specify the location of the data root in an environment file.

```shell
# example: predicting antibody affinity on a CoV-SARS-2 epitope with ESM2 embeddings
python -m immune_embeddings.pipelines --config experiments/covid_affinity_esm/covid_exp_650M_6.json
```

## Data

We do not distribute the original data used for our experiments. We however provide the directory structure of a data
directory as the `data` directory, where text files specify where to obtain data and how to format it.