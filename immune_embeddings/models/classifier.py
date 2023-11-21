from typing import Dict, Callable, Union

import torch
from torch import nn

from immune_embeddings.models import BaseModel


class Classifier(BaseModel):
    def __init__(self, *, model: nn.Module, loss: nn.Module = None, optimization: dict = None,
                 metrics: Union[None, Dict[str, Union[str, Callable]]] = None, predict_probas: bool = False,
                 proba_activation: nn.Module, input_column: str = "X", label_column: str = "Y"):
        loss = loss or nn.NLLLoss()
        super(Classifier, self).__init__(metrics=metrics, loss=loss, optimization=optimization)
        self.model = model

        self.predict_probas = predict_probas
        self.proba_activation = proba_activation
        if predict_probas and proba_activation is None:
            self.proba_activation = nn.Softmax(dim=-1)
        self.input_column = input_column
        self.label_column = label_column

    def step(self, batch, phase):
        X = batch[self.input_column]
        y = batch[self.label_column]
        preds = self.model(X)
        self.log_step_metrics(preds, y, phase)
        return self.compute_loss(preds, y.type(preds.type()), phase=phase)

    def predict_step(self, batch, batch_idx, *args):
        X = batch[self.input_column]
        preds = self.model(X)
        if self.predict_probas:
            return self.proba_activation(preds)
        else:
            return torch.argmax(preds, dim=-1)


class BinaryClassifier(Classifier):
    def __init__(self, *, model: nn.Module, loss: nn.Module = None, optimization: dict = None,
                 metrics: Union[None, Dict[str, Union[str, Callable]]] = None, predict_probas: bool = True,
                 proba_activation: nn.Module = None, input_column: str = "X", label_column: str = "Y"):
        loss = loss or nn.BCELoss()
        proba_activation = proba_activation or nn.Identity()

        super(BinaryClassifier, self).__init__(model=model, loss=loss, optimization=optimization, metrics=metrics,
                                               predict_probas=predict_probas, proba_activation=proba_activation,
                                               input_column=input_column, label_column=label_column)

    def predict_step(self, batch, batch_idx, *args):
        X = batch[self.input_column]
        preds = self.model(X)
        if self.predict_probas:
            return preds
        else:
            return (preds >= 0.5).to(torch.int)
