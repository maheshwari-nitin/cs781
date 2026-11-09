"""Architecture C: Gated DeltaNet hybrid (Qwen3-Next style). Your implementation: see models/README.md for the interface and spec."""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from fla.layers import GatedDeltaNet
except ImportError:
    GatedDeltaNet = None

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


class GDNStateCache:
    """Minimal stand-in for fla's Cache object (avoids its dependence on the transformers version).
    fla's GatedDeltaNet reads past_key_values[layer_idx] -> dict(recurrent_state, conv_state) and calls
    past_key_values.update(layer_idx=..., recurrent_state=..., conv_state=..., offset=...).
    GDN layers must have consecutive layer_idx 0..n_gdn-1 (entries are appended in order)."""

    def __init__(self):
        self.states = []

    def __len__(self):
        return len(self.states)

    def __getitem__(self, layer_idx):
        return self.states[layer_idx]

    def update(self, layer_idx=0, recurrent_state=None, conv_state=None, offset=1, **kwargs):
        if len(self.states) <= layer_idx:
            self.states.append(dict(recurrent_state=recurrent_state, conv_state=conv_state))
        else:
            state = self.states[layer_idx]
            if recurrent_state is not None:
                state["recurrent_state"] = recurrent_state
            if conv_state is not None:
                state["conv_state"] = conv_state
        return self.states[layer_idx]

def precompute_freqs_cis(dim: int, end: int, theta: float = 100000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def apply_rotary_emb(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
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

class GatedAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.g_proj = nn.Linear(dim, dim, bias=False)  # Output gate
        
        # QK-Norm per head before RoPE
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, kv_cache=None):
        B, T, _ = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
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

        if self.n_heads != self.n_kv_heads:
            num_repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(num_repeat, dim=1)
            v = v.repeat_interleave(num_repeat, dim=1)

        # Training / prefill (T>1, empty cache): causal mask. Cached decode step (T=1): attend to all keys.
        is_causal = T > 1
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        
        # Apply output projection and sigmoid gate
        o = self.o_proj(attn_out)
        gate = torch.sigmoid(self.g_proj(x))
        return o * gate, kv_cache

class AttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int, ffn_dim: int):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GatedAttention(d_model, n_heads, n_kv_heads, head_dim)
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = MLP(d_model, ffn_dim)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, kv_cache=None):
        h, new_kv = self.attn(self.attn_norm(x), freqs_cos, freqs_sin, kv_cache)
        x = x + h
        x = x + self.mlp(self.mlp_norm(x))
        return x, new_kv

class GDNBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int, expand_v: float, ffn_dim: int, layer_idx: int):
        super().__init__()
        if GatedDeltaNet is None:
            raise RuntimeError("flash-linear-attention (fla) is not installed. Please pip install flash-linear-attention.")

        self.attn_norm = RMSNorm(d_model)
        # fla's argument is `hidden_size` (an unknown `d_model=` is silently swallowed by **kwargs and the
        # layer falls back to hidden_size=2048, head_dim=256). layer_idx indexes this layer's recurrent/conv
        # state in the cache during generation.
        self.mixer = GatedDeltaNet(hidden_size=d_model, num_heads=n_heads, head_dim=head_dim,
                                   expand_v=expand_v, layer_idx=layer_idx)
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = MLP(d_model, ffn_dim)

    def forward(self, x: torch.Tensor, cache: "GDNStateCache" = None):
        norm_x = self.attn_norm(x)
        # fla returns (output, attentions=None, past_key_values); the cache object is updated in place
        out = self.mixer(norm_x, past_key_values=cache, use_cache=cache is not None)
        mixer_out = out[0] if isinstance(out, tuple) else out

        x = x + mixer_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, cache

class HybridGDNModel(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int = 1024, 
                 n_layers: int = 24, n_heads: int = 16, n_kv_heads: int = 4,
                 head_dim: int = 64, ffn_dim: int = 1920, gdn_head_dim: int = 64, gdn_expand_v: float = 1.0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.attn_dim = n_heads * head_dim

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList()
        gdn_idx = 0
        for i in range(n_layers):
            # 3 GDN : 1 Attn pattern
            if (i + 1) % 4 == 0:
                self.layers.append(AttnBlock(d_model, n_heads, n_kv_heads, head_dim, ffn_dim))
            else:
                self.layers.append(GDNBlock(d_model, n_heads, gdn_head_dim, gdn_expand_v, ffn_dim, layer_idx=gdn_idx))
                gdn_idx += 1
        self.n_attn_layers = n_layers - gdn_idx
                
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie embeddings
        self.output.weight = self.tok_embeddings.weight
        
        # Precompute RoPE frequencies for the attention layers
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, max_seq_len, theta=100000.0)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # Linear/Embedding weights ~ N(0, 0.02); fla's special params (A_log, dt_bias, short convs) keep
        # their own init since they are not nn.Linear/nn.Embedding
        self.apply(self._init_weights)
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
            if isinstance(layer, AttnBlock):
                x, _ = layer(x, freqs_cos, freqs_sin, kv_cache=None)
            else:
                x, _ = layer(x)  # no cache during training
                
        x = self.norm(x)
        logits = self.output(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss

    def flops_per_token(self, seq_len: int) -> float:
        # 6N for the matmuls + softmax-attention cost of the attention layers only (12*L_attn*d_attn*T);
        # the GDN chunked recurrence is linear in T and small relative to its projections
        params = sum(p.numel() for p in self.parameters())
        return 6.0 * params + 12.0 * self.n_attn_layers * self.attn_dim * seq_len

    @torch.no_grad()
    def generate_tokens(self, idx: torch.Tensor, max_new: int, sample_fn) -> torch.Tensor:
        gdn_cache = GDNStateCache()                 # recurrent + conv states of all GDN layers
        attn_caches = [None] * len(self.layers)     # (k, v) per attention layer
        out_tokens = []
        current_idx = idx
        start_pos = 0                               # absolute position of current_idx[:, 0] (for RoPE)

        for i in range(max_new):
            B, T = current_idx.shape
            freqs_cos = self.freqs_cos[start_pos : start_pos + T]
            freqs_sin = self.freqs_sin[start_pos : start_pos + T]

            x = self.tok_embeddings(current_idx)
            for j, layer in enumerate(self.layers):
                if isinstance(layer, AttnBlock):
                    x, attn_caches[j] = layer(x, freqs_cos, freqs_sin, attn_caches[j])
                else:
                    x, _ = layer(x, gdn_cache)
            start_pos += T
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
    n_layers = kw.get("n_layers", 24)
    n_heads = kw.get("n_heads", 16)
    n_kv_heads = kw.get("n_kv_heads", 4)
    head_dim = kw.get("head_dim", 64)
    # ffn 1920 (not 2560) keeps the model under 300M: a GDN layer (q, k, v, gate, o projections) is ~5.3M
    # params vs ~2.6M for a GQA attention layer, so 24 layers with ffn 2560 would be ~338M
    ffn_dim = kw.get("ffn_dim", 1920)
    gdn_head_dim = kw.get("gdn_head_dim", 64)
    gdn_expand_v = kw.get("gdn_expand_v", 1.0)

    return HybridGDNModel(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        ffn_dim=ffn_dim,
        gdn_head_dim=gdn_head_dim,
        gdn_expand_v=gdn_expand_v
    )