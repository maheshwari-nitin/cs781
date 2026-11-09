import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_TYPE = "decoder"

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
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        
        # QK-Norm per head before RoPE
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, kv_cache=None):
        B, T, _ = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        
        # Apply QK-Norm per head (operates on the last dim: head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Transpose to [B, H, T, D] for RoPE and Attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        kv_cache = (k, v)  # always return the cache so the prefill call starts one

        # Grouped Query Attention: repeat K and V to match Q's heads
        if self.n_heads != self.n_kv_heads:
            num_repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(num_repeat, dim=1)
            v = v.repeat_interleave(num_repeat, dim=1)
            
        # Training / prefill (T>1, empty cache): causal mask. Cached decode step (T=1): attend to all keys.
        is_causal = T > 1
        
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        
        return self.o_proj(attn_out), kv_cache

class QwenBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int, ffn_dim: int):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, n_kv_heads, head_dim)
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = MLP(d_model, ffn_dim)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, kv_cache=None):
        h, new_kv = self.attn(self.attn_norm(x), freqs_cos, freqs_sin, kv_cache)
        x = x + h
        x = x + self.mlp(self.mlp_norm(x))
        return x, new_kv

class QwenDenseModel(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int = 1024, 
                 n_layers: int = 22, n_heads: int = 16, n_kv_heads: int = 4, 
                 head_dim: int = 64, ffn_dim: int = 2816):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.attn_dim = n_heads * head_dim

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            QwenBlock(d_model, n_heads, n_kv_heads, head_dim, ffn_dim)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie embeddings
        self.output.weight = self.tok_embeddings.weight
        
        # Precompute RoPE frequencies
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, max_seq_len, theta=100000.0)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.apply(self._init_weights)
        # scale down residual-branch output projections (GPT-2 style) so the residual stream
        # variance doesn't grow with depth
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * n_layers) ** 0.5)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        freqs_cos = self.freqs_cos[:T]
        freqs_sin = self.freqs_sin[:T]
        
        x = self.tok_embeddings(idx)
        
        for layer in self.layers:
            x, _ = layer(x, freqs_cos, freqs_sin, kv_cache=None)
            
        x = self.norm(x)
        logits = self.output(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            
        return logits, loss

    def flops_per_token(self, seq_len: int) -> float:
        # 6N for the matmuls (forward + backward) + 12*L*d_attn*T for attention scores/values (PaLM appendix B)
        params = sum(p.numel() for p in self.parameters())
        return 6.0 * params + 12.0 * self.n_layers * self.attn_dim * seq_len

    @torch.no_grad()
    def generate_tokens(self, idx: torch.Tensor, max_new: int, sample_fn) -> torch.Tensor:
        kv_caches = None
        out_tokens = []
        current_idx = idx
        
        for i in range(max_new):
            B, T = current_idx.shape
            start_pos = 0 if kv_caches is None else kv_caches[0][0].size(2)
            
            freqs_cos = self.freqs_cos[start_pos : start_pos + T]
            freqs_sin = self.freqs_sin[start_pos : start_pos + T]
            
            x = self.tok_embeddings(current_idx)
            new_caches = []
            
            for j, layer in enumerate(self.layers):
                kv = kv_caches[j] if kv_caches is not None else None
                h = layer.attn_norm(x)
                h, new_kv = layer.attn(h, freqs_cos, freqs_sin, kv)
                new_caches.append(new_kv)
                x = x + h
                x = x + layer.mlp(layer.mlp_norm(x))
                
            kv_caches = new_caches
            x = self.norm(x)
            logits = self.output(x)
            
            next_token_logits = logits[:, -1, :]
            next_token = sample_fn(next_token_logits, idx)
            out_tokens.append(next_token)
            
            current_idx = next_token.unsqueeze(1)
            idx = torch.cat([idx, current_idx], dim=1)
            
        return torch.stack(out_tokens, dim=1)


def build_model(vocab_size: int, max_seq_len: int, **kw) -> nn.Module:
    d_model = kw.get("d_model", 1024)
    n_layers = kw.get("n_layers", 22)
    n_heads = kw.get("n_heads", 16)
    n_kv_heads = kw.get("n_kv_heads", 4)
    head_dim = kw.get("head_dim", 64)
    ffn_dim = kw.get("ffn_dim", 2816)
    
    return QwenDenseModel(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        ffn_dim=ffn_dim
    )