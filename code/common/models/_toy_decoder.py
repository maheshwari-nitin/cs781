"""SMOKE TEST ONLY: a tiny decoder built from torch built-ins, used to check the pipeline end to end.
Not a submission model (below the 50M floor; the trainer only enforces the budget on GPU)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_TYPE = "decoder"


class ToyDecoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d=128, layers=2, heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(max_seq_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.emb.weight
        for e in (self.emb, self.pos):
            nn.init.normal_(e.weight, std=0.02)

    def forward(self, idx, targets=None):
        T = idx.size(1)
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        logits = self.head(self.body(x, mask=mask, is_causal=True))
        loss = None if targets is None else F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def build_model(vocab_size, max_seq_len, **kw):
    return ToyDecoder(vocab_size, max_seq_len, **kw)
