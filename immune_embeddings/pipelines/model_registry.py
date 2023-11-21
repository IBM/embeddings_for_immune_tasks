import inspect
from types import ModuleType

import pytorch_lightning as pl
from torch import nn

from immune_embeddings.models import classifier
from immune_embeddings.models import embeddings
from immune_embeddings.models import mlp
from immune_embeddings.models import regressor
from immune_embeddings.pipelines.registry import ConstructorRegistry


class ModelConstructorRegistry(ConstructorRegistry):
    def __init__(self):
        super(ModelConstructorRegistry, self).__init__(name_key="model", param_key="model_params")
        model_classes = [nn.Module, pl.LightningModule]
        self.model_classes = tuple(model_classes)

    def get(self, params):
        submodels = dict()
        if "submodels" in params:
            for submodel_name, submodel_spec in params["submodels"].items():
                submodels[submodel_name] = self.get(submodel_spec)
            params[self.param_key].update(submodels)

        return super(ModelConstructorRegistry, self).get(params=params)

    def auto_register(self, model_module: ModuleType):
        module_classes = inspect.getmembers(model_module, inspect.isclass)

        for class_name, class_object in module_classes:
            if issubclass(class_object, self.model_classes):
                self.register(class_name, class_object)


model_registry = ModelConstructorRegistry()
for m in [embeddings, classifier, mlp, nn]:
    model_registry.auto_register(m)
model_registry.entry(regressor.Regressor)
