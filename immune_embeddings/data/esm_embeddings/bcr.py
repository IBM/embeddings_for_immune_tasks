from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

from torch.utils.data import Dataset

from immune_embeddings.data.data_utils import SplitDataModule
from immune_embeddings.data.esm_embeddings import ESMDataDir


@dataclass
class BCellESMDataDir(ESMDataDir):
    data_relpath: Path = Path("bcr") / "synthetic" / "esm_mean_embeddings"


class DictDataset(Dataset):
    def __init__(self, prediction_dict: Dict[str, Any], drop_keys=("protein_tokens",)):
        self.prediction_dict = {k: prediction_dict[k] for k in prediction_dict if k not in drop_keys}

    def __getitem__(self, item):
        return {k: self.prediction_dict[k][item] for k in self.prediction_dict}

    def __len__(self):
        lens = set([len(self.prediction_dict[k]) for k in self.prediction_dict])
        assert len(lens) == 1
        return lens.pop()

    def keys(self):
        return self.prediction_dict.keys()


class SplitDataModulePredictAll(SplitDataModule):
    def predict_dataloader(self):
        return [self.train_dataloader(), self.val_dataloader(), self.test_dataloader()]


def get_bcr_embeddings(batch_size, *, model_spec="esm2_t6_8M_UR50D", data_root=None, override_env_vars=False,
                       usecwd=False, random_seed=1234, num_workers=0):
    data_dir = BCellESMDataDir(
        data_root=data_root, override_env_vars=override_env_vars, usecwd=usecwd)

    esm_data = data_dir.get_predictions(model_spec)
    train_set = DictDataset(esm_data["train"])
    val_set = DictDataset(esm_data["val"])
    test_set = DictDataset(esm_data["test"])

    dm = SplitDataModulePredictAll(batch_size, train_set=train_set, valid_set=val_set, test_set=test_set,
                                   random_seed=random_seed, num_workers=num_workers)

    return dm
