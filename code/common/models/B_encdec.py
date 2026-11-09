"""Architecture B: encoder-decoder. Your implementation: see models/README.md for the interface and spec."""
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_TYPE = "encdec"

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * norm * self.weight).to(dtype)

def precompute_freqs_cis(dim: int, end: int, theta: float = 100000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def apply_rotary_emb(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    # x shape: [B, H, T, D]
    # freqs_cos, freqs_sin shape: [T, D/2]
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(0)
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(0)
    
    x_r, x_i = x.float().view(*x.shape[:-1], -1, 2).unbind(-1)
    x_out = torch.stack([
        x_r * freqs_cos - x_i * freqs_sin,
        x_r * freqs_sin + x_i * freqs_cos
    ], dim=-1).flatten(3)
    return x_out.type_as(x)

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))

class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, apply_rope: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.apply_rope = apply_rope
        
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        
        # QK-Norm per head
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x: torch.Tensor, memory: torch.Tensor = None, freqs_cos: torch.Tensor = None, freqs_sin: torch.Tensor = None, is_causal: bool = False, kv_cache=None):
        B, T, _ = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        q = self.q_norm(q)
        q = q.transpose(1, 2)
        
        is_cross_attn = memory is not None
        src = memory if is_cross_attn else x
        _, S, _ = src.shape
        
        if is_cross_attn and kv_cache is not None:
            # Cross-attention cache: K and V of memory are static
            k, v = kv_cache
        else:
            k = self.k_proj(src).view(B, S, self.n_kv_heads, self.head_dim)
            v = self.v_proj(src).view(B, S, self.n_kv_heads, self.head_dim)
            k = self.k_norm(k)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        
        if self.apply_rope and freqs_cos is not None:
            q = apply_rotary_emb(q, freqs_cos, freqs_sin)
            if not is_cross_attn:
                k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        
        if kv_cache is not None and not is_cross_attn:
            # Self-attention cache update
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
            kv_cache = (k, v)
        elif is_cross_attn and kv_cache is None:
            # Initialize cross-attention cache
            kv_cache = (k, v)
            
        if self.n_heads != self.n_kv_heads:
            num_repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(num_repeat, dim=1)
            v = v.repeat_interleave(num_repeat, dim=1)
            
        causal_mask = is_causal and (kv_cache is None or is_cross_attn)
        if kv_cache is not None and not is_cross_attn and T == 1:
            causal_mask = False
            
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=causal_mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        
        return self.o_proj(attn_out), kv_cache

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int, ffn_dim: int):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, n_kv_heads, head_dim, apply_rope=True)
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = MLP(d_model, ffn_dim)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        h, _ = self.attn(self.attn_norm(x), freqs_cos=freqs_cos, freqs_sin=freqs_sin, is_causal=False)
        x = x + h
        x = x + self.mlp(self.mlp_norm(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int, ffn_dim: int):
        super().__init__()
        self.self_attn_norm = RMSNorm(d_model)
        self.self_attn = Attention(d_model, n_heads, n_kv_heads, head_dim, apply_rope=True)
        
        self.cross_attn_norm = RMSNorm(d_model)
        # Cross-attention typically doesn't use RoPE as sequence spaces differ
        self.cross_attn = Attention(d_model, n_heads, n_kv_heads, head_dim, apply_rope=False)
        
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = MLP(d_model, ffn_dim)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        h, _ = self.self_attn(self.self_attn_norm(x), freqs_cos=freqs_cos, freqs_sin=freqs_sin, is_causal=True)
        x = x + h
        
        h, _ = self.cross_attn(self.cross_attn_norm(x), memory=memory, is_causal=False)
        x = x + h
        
        x = x + self.mlp(self.mlp_norm(x))
        return x

class EncDecModel(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int = 1024, 
                 n_enc_layers: int = 10, n_dec_layers: int = 10, n_heads: int = 16, 
                 n_kv_heads: int = 4, head_dim: int = 64, ffn_dim: int = 2816):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.n_enc_layers, self.n_dec_layers = n_enc_layers, n_dec_layers
        self.attn_dim = n_heads * head_dim

        self.shared_embeddings = nn.Embedding(vocab_size, d_model)
        
        self.encoder = nn.ModuleList([
            EncoderBlock(d_model, n_heads, n_kv_heads, head_dim, ffn_dim)
            for _ in range(n_enc_layers)
        ])
        self.enc_norm = RMSNorm(d_model)
        
        self.decoder = nn.ModuleList([
            DecoderBlock(d_model, n_heads, n_kv_heads, head_dim, ffn_dim)
            for _ in range(n_dec_layers)
        ])
        self.dec_norm = RMSNorm(d_model)
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.shared_embeddings.weight
        
        # Precompute RoPE frequencies
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, max_seq_len, theta=100000.0)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.apply(self._init_weights)
        # scale down residual-branch output projections by the number of residual additions in each stack
        # (encoder: 2 per layer; decoder: 3 per layer -- self-attn, cross-attn, MLP)
        for prefix, n_res in (("encoder.", 2 * n_enc_layers), ("decoder.", 3 * n_dec_layers)):
            for name, p in self.named_parameters():
                if name.startswith(prefix) and (name.endswith("o_proj.weight") or name.endswith("down.weight")):
                    nn.init.normal_(p, mean=0.0, std=0.02 / n_res ** 0.5)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        S = src.size(1)
        freqs_cos = self.freqs_cos[:S]
        freqs_sin = self.freqs_sin[:S]
        
        x = self.shared_embeddings(src)
        for layer in self.encoder:
            x = layer(x, freqs_cos, freqs_sin)
            
        return self.enc_norm(x)

    def decode(self, memory: torch.Tensor, dec_in: torch.Tensor) -> torch.Tensor:
        L = dec_in.size(1)
        freqs_cos = self.freqs_cos[:L]
        freqs_sin = self.freqs_sin[:L]
        
        x = self.shared_embeddings(dec_in)
        for layer in self.decoder:
            x = layer(x, memory, freqs_cos, freqs_sin)
            
        x = self.dec_norm(x)
        return self.lm_head(x)

    def forward(self, src: torch.Tensor, dec_in: torch.Tensor, targets: torch.Tensor = None):
        memory = self.encode(src)
        logits = self.decode(memory, dec_in)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            
        return logits, loss

    def flops_per_token(self, seq_len: int, src_len: int = 768) -> float:
        # Training FLOPs per *window* token (the trainer counts src + tgt tokens). The encoder only runs on
        # the S source tokens and the decoder + LM head only on the L target tokens, so plain 6N would
        # over-count by ~2x. Attention adds 12 * layers * d_attn * (keys attended) per query token.
        S, L = src_len, seq_len - src_len
        n_enc = sum(p.numel() for p in self.encoder.parameters())
        n_dec = sum(p.numel() for p in self.decoder.parameters()) + self.lm_head.weight.numel()
        matmul = 6.0 * (n_enc * S + n_dec * L)
        attn = 12.0 * self.attn_dim * (self.n_enc_layers * S * S + self.n_dec_layers * L * (L + S))
        return (matmul + attn) / seq_len


def build_model(vocab_size: int, max_seq_len: int, **kw) -> nn.Module:
    d_model = kw.get("d_model", 1024)
    n_enc_layers = kw.get("n_enc_layers", 10)
    n_dec_layers = kw.get("n_dec_layers", 10)
    n_heads = kw.get("n_heads", 16)
    n_kv_heads = kw.get("n_kv_heads", 4)
    head_dim = kw.get("head_dim", 64)
    ffn_dim = kw.get("ffn_dim", 2816)
    
    return EncDecModel(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_enc_layers=n_enc_layers,
        n_dec_layers=n_dec_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        ffn_dim=ffn_dim
    )