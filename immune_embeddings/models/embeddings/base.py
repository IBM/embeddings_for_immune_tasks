from abc import ABC

import pytorch_lightning as pl
import torch

from immune_embeddings.models import DictBasedPredictor


class EmbeddingModel(torch.nn.Module, ABC):
    """Abstract class for embeddings"""

    def __init__(self, embedder: torch.nn.Module, freeze_weights=True):
        super(EmbeddingModel, self).__init__()
        self.embedder = embedder
        if freeze_weights:
            for param in self.parameters():
                param.requires_grad = False

    def predict(self, data: dict):
        self.eval()
        with torch.no_grad():
            return self(data)


class EmbeddingPredictor(DictBasedPredictor, pl.LightningModule):
    def __init__(self, embedding: EmbeddingModel, pass_sequence=True):
        super(EmbeddingPredictor, self).__init__()
        self.embbeding = embedding
        self.pass_sequence = pass_sequence

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        representations = self.embbeding.predict(batch)
        batch["representations"] = representations
        if not self.pass_sequence:
            del batch["protein_sequences"]
        return batch
