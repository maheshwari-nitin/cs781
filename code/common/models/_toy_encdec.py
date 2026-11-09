"""SMOKE TEST ONLY: a tiny encoder-decoder from torch built-ins (nn.Transformer). Not a submission model."""
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_TYPE = "encdec"


class ToyEncDec(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d=128, layers=1, heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(max_seq_len, d)
        self.tf = nn.Transformer(d, heads, layers, layers, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.emb.weight
        for e in (self.emb, self.pos):
            nn.init.normal_(e.weight, std=0.02)

    def _embed(self, x):
        return self.emb(x) + self.pos(torch.arange(x.size(1), device=x.device))

    def encode(self, src):
        return self.tf.encoder(self._embed(src))

    def decode(self, memory, dec_in):
        L = dec_in.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=dec_in.device)
        return self.head(self.tf.decoder(self._embed(dec_in), memory, tgt_mask=mask, tgt_is_causal=True))

    def forward(self, src, dec_in, targets=None):
        logits = self.decode(self.encode(src), dec_in)
        loss = None if targets is None else F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def build_model(vocab_size, max_seq_len, **kw):
    return ToyEncDec(vocab_size, max_seq_len, **kw)
