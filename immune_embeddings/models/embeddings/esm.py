import esm
import torch

from immune_embeddings.models.embeddings.base import EmbeddingModel


class ESMEmbeddingModel(EmbeddingModel):
    """Meta ESM protein sequence embeddings"""

    embedder: esm.ESM2

    def __init__(self, cache_dir=None, model_id="esm2_t33_650M_UR50D", repr_layer=33, fixed_size=True,
                 freeze_weights=True):
        if cache_dir is not None:
            torch.hub.set_dir(cache_dir)
        model, alphabet = torch.hub.load("facebookresearch/esm:main", model_id)

        super(ESMEmbeddingModel, self).__init__(embedder=model, freeze_weights=freeze_weights)
        self.model_id = model_id
        self.repr_layer = repr_layer
        self.fixed_size = fixed_size
        self.alphabet = alphabet

    def forward(self, data: dict):
        batch_tokens = data["protein_tokens"]

        results = self.embedder(batch_tokens, repr_layers=[self.repr_layer])
        representations = results["representations"][self.repr_layer]

        if self.fixed_size:
            protein_lens = data["protein_lens"]
            representations = self.avg_sequences(representations, protein_lens)
            results["representations"][self.repr_layer] = representations

        return representations

    @staticmethod
    def avg_sequences(token_representations: torch.Tensor, protein_lens: torch.Tensor):

        # drop the starting <cls> token
        token_representations = token_representations[:, 1:]
        cumuls = token_representations.cumsum(1)

        # We need to tokenize to be robust to multi-character tokens like <mask>
        # get cumuls[i,protein_lens[i]] and normalize
        sequence_embeddings = cumuls[torch.arange(cumuls.shape[0]), protein_lens] / protein_lens.unsqueeze(1)

        return sequence_embeddings
