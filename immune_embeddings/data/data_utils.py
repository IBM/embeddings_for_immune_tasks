import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class SplitDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_set: Dataset, valid_set: Dataset = None, test_set: Dataset = None,
                 test_fraction: float = 0.2, valid_fraction: float = 0.2, random_seed=1234, num_workers=None, ):
        super(SplitDataModule, self).__init__()

        self.batch_size = batch_size
        self.train = train_set
        self.valid = valid_set
        self.test = test_set
        self.generator = torch.Generator().manual_seed(random_seed)

        self.valid_frac = valid_fraction
        self.test_frac = test_fraction
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if self.test is None:
            self.train, self.test = self.do_split(self.train, self.test_frac)

        if self.valid is None:
            self.train, self.valid = self.do_split(self.train, self.valid_frac)

    def do_split(self, train_set, fraction):
        train_len = len(train_set)
        other_len = int(fraction * train_len)
        train_len = train_len - other_len
        return random_split(self.train, [train_len, other_len], generator=self.generator)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          generator=self.generator)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)


class TabularDataset(Dataset):
    """Tabular dataset from a csv file"""

    def __init__(self, dataframe, input_columns, target_columns,
                 input_dtype=np.float32, map_values=None):
        df = dataframe
        if map_values is not None:
            df = df.replace(map_values)
        self.X = df[input_columns].to_numpy(dtype=input_dtype)

        if len(target_columns) == 1:
            target_columns = target_columns[0]

        self.y = df[target_columns].to_numpy()
        assert len(self.y) == len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return len(self.X)
