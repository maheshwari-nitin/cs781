"""Document classifier for long judgments: every chunk goes through the Phase-1 backbone (features from
lm_lib.ner.model.hidden_states), token states are pooled into one vector per chunk, chunk vectors are aggregated
into one document vector, then a classification head. Has .backbone and .head, so the adapters/ methods work."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ner.model import hidden_states


class DocClassifier(nn.Module):
    def __init__(self, backbone, n_classes=2, token_pooling="mean", chunk_agg="mean", head=None, dropout=0.1):
        super().__init__()
        self.backbone, self.token_pooling, self.chunk_agg = backbone, token_pooling, chunk_agg
        with torch.no_grad():
            dev = next(backbone.parameters()).device
            was = backbone.training
            backbone.eval()
            self.hidden_dim = hidden_states(backbone, torch.zeros(1, 8, dtype=torch.long, device=dev)).size(-1)
            backbone.train(was)
        if chunk_agg == "attention":                    # learned query scoring each chunk vector
            self.chunk_query = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.head = head if head is not None else nn.Linear(self.hidden_dim, n_classes)

    def chunk_vectors(self, ids, tok_mask):
        """ids/tok_mask [N, L] -> [N, D] (one vector per chunk)."""
        h = hidden_states(self.backbone, ids).float()
        m = tok_mask.unsqueeze(-1).float()
        if self.token_pooling == "mean":
            return (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        if self.token_pooling == "last":                 # last real token (right padding)
            last = tok_mask.long().sum(1).clamp(min=1) - 1
            return h[torch.arange(h.size(0), device=h.device), last]
        raise ValueError(self.token_pooling)

    def forward(self, ids, tok_mask, chunk_mask):
        """ids/tok_mask [B, C, L]; chunk_mask [B, C] (True = real chunk) -> logits [B, n_classes]."""
        B, C, L = ids.shape
        v = self.chunk_vectors(ids.view(B * C, L), tok_mask.view(B * C, L)).view(B, C, -1)
        cm = chunk_mask.unsqueeze(-1).float()
        if self.chunk_agg == "mean":
            doc = (v * cm).sum(1) / cm.sum(1).clamp(min=1.0)
        elif self.chunk_agg == "max":
            doc = v.masked_fill(cm == 0, float("-inf")).max(1).values
        elif self.chunk_agg == "attention":
            w = self.chunk_query(v).squeeze(-1).masked_fill(~chunk_mask, float("-inf"))
            doc = (F.softmax(w, -1).unsqueeze(-1) * v).sum(1)
        else:
            raise ValueError(self.chunk_agg)
        return self.head(self.dropout(doc))
