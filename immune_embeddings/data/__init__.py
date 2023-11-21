import os
from dataclasses import dataclass
from pathlib import Path

import dotenv


def get_data_root(data_root=None, override_env_vars=False, usecwd=False):
    if data_root is not None:
        data_root = Path(data_root).absolute()
        assert data_root.exists()
        return data_root

    if not override_env_vars:
        try:
            data_root = Path(os.environ["IMMUNE_EMBEDDINGS_DATA_ROOT"])
            assert data_root.exists()
            return data_root
        except KeyError:
            pass

    dotenv_path = dotenv.find_dotenv(usecwd=usecwd)
    dotenv_config = dotenv.dotenv_values(dotenv_path)
    os.environ["IMMUNE_EMBEDDINGS_DATA_ROOT"] = dotenv_config["IMMUNE_EMBEDDINGS_DATA_ROOT"]

    data_root = Path(os.environ["IMMUNE_EMBEDDINGS_DATA_ROOT"])
    assert data_root.exists()
    return data_root


@dataclass
class DataDir:
    data_relpath: Path
    data_root: Path = None
    override_env_vars: bool = False
    usecwd: bool = False

    def __post_init__(self):
        if self.data_root is None:
            self.set_data_root(data_root=self.data_root, override_env_vars=self.override_env_vars, usecwd=self.usecwd)

    def set_data_root(self, data_root=None, override_env_vars=None, usecwd=None):
        self.data_root = get_data_root(data_root=data_root, override_env_vars=override_env_vars, usecwd=usecwd)

    @property
    def data_dir(self):
        data_dir = self.data_root / self.data_relpath
        assert data_dir.exists()
        return data_dir
