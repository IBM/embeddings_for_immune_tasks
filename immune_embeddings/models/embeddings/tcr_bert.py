from typing import Any, Dict

from pytorch_lightning import LightningModule
from sentence_transformers import SentenceTransformer


class TCRBertEmbedder(LightningModule):
    def __init__(self):
        super(TCRBertEmbedder, self).__init__()
        self.tcr_bert = SentenceTransformer("wukevin/tcr-bert-mlm-only")

    def predict_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> Any:
        embedding = self.tcr_bert(batch)["sentence_embedding"]
        return {"sequence_embedding": embedding, "sequence_idx": "sequence_idx"}
