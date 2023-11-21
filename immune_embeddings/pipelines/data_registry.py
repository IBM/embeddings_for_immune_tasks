from immune_embeddings.data.bcr_guikema_to_esm import get_bcr_guikema_for_esm
from immune_embeddings.data.covid_regression_from_ablang import (get_covid_ablang_datamodule)
from immune_embeddings.data.covid_regression_from_esm import get_covid_esm_datamodule
from immune_embeddings.data.covid_to_esm import get_covid_for_esm
from immune_embeddings.data.esm_embeddings.bcr import get_bcr_embeddings
from immune_embeddings.data.titan_classification import (get_titan_embedding_classification_datamodule)
from immune_embeddings.pipelines.registry import ConstructorRegistry

data_registry = ConstructorRegistry(name_key="dataset", param_key="dataset_params")

data_registry.register("bcell_embeddings", get_bcr_embeddings)
data_registry.register("covid_affinity_ablang", get_covid_ablang_datamodule)
data_registry.register("covid_affinity_esm", get_covid_esm_datamodule)
data_registry.register("covid_to_esm", get_covid_for_esm)
data_registry.register("guikema_to_esm", get_bcr_guikema_for_esm)
data_registry.register("titan_classification", get_titan_embedding_classification_datamodule)
