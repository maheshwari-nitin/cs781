"""Sentence classifier on a Phase-1 backbone: token features (lm_lib.ner.model.hidden_states) are pooled over
each sentence's token range, then classified. Has .backbone and .head, so lm_lib.ner.model.apply_method and the
adapters/ methods (LoRMA, LoRA, ...) work unchanged."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ner.model import hidden_states


class SentenceClassifier(nn.Module):
    def __init__(self, backbone, n_labels, pooling="mean", head=None, dropout=0.1):
        super().__init__()
        self.backbone, self.pooling = backbone, pooling
        with torch.no_grad():
            dev = next(backbone.parameters()).device
            was = backbone.training
            backbone.eval()
            self.hidden_dim = hidden_states(backbone, torch.zeros(1, 8, dtype=torch.long, device=dev)).size(-1)
            backbone.train(was)
        self.dropout = nn.Dropout(dropout)
        self.head = head if head is not None else nn.Linear(self.hidden_dim, n_labels)

    def pool(self, h, segs):
        """h [B, T, D]; segs Long [N, 3] = (batch index, start, end) -> [N, D]."""
        b, s, e = segs[:, 0], segs[:, 1], segs[:, 2]
        if self.pooling == "last":                       # last token of the sentence (has read all of it)
            return h[b, e - 1]
        if self.pooling == "mean":
            cs = F.pad(h.float().cumsum(1), (0, 0, 1, 0))  # [B, T+1, D] prefix sums
            return (cs[b, e] - cs[b, s]) / (e - s).unsqueeze(-1).float()
        raise ValueError(self.pooling)

    def forward(self, ids, segs):
        return self.head(self.dropout(self.pool(hidden_states(self.backbone, ids), segs)))
