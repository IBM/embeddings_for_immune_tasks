import json
import re
from dataclasses import dataclass
from typing import Dict

import torch

from immune_embeddings.data import DataDir


@dataclass
class ESMDataDir(DataDir):
    @property
    def model_paths(self):
        source_pattern = re.compile("esm2_t\d+_\d+[MB]_UR50D")
        return [f for f in self.data_dir.glob("*") if source_pattern.match(f.name)]

    @property
    def model_ids(self):
        return [m.name for m in self.model_paths]

    @property
    def nparams_to_model_id(self) -> Dict[str, str]:
        id_pattern = re.compile("esm2_t\d+_(\d+[MB])_UR50D")
        models_nparams = dict()
        for model_id in self.model_ids:
            n_params = id_pattern.search(model_id).group(1)
            models_nparams[n_params] = model_id

        return models_nparams

    def get_model_id(self, param_spec):
        return self.nparams_to_model_id[param_spec]

    def get_model(self, query):
        if query in self.model_ids:
            return query
        if query in self.nparams_to_model_id:
            return self.nparams_to_model_id[query]
        raise KeyError(f"Query {query} does not specify and available embedding")

    def get_file(self, query, filename):
        model_name = self.get_model(query)
        return self.data_dir / model_name / filename

    def get_raw_config(self, query):
        with open(self.get_file(query, "raw_config.json"), "r") as f:
            return json.load(f)

    def get_config(self, query):
        with open(self.get_file(query, "processed_config.json"), "r") as f:
            return json.load(f)

    def get_predictions(self, query):
        fpath = self.get_file(query, "predictions.pt")
        return torch.load(fpath)
