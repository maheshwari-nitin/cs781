"""Architecture D: DeepSeek-style MLA + MoE. Your implementation: see models/README.md for the interface and spec."""
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

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def apply_rotary_emb(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    # x shape: [B, T, H, D] or [B, T, 1, D]
    # freqs_cos, freqs_sin shape: [T, D/2]
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2) # [1, T, 1, D/2]
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2)
    
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

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, q_lora_rank: int, kv_lora_rank: int, 
                 qk_nope_dim: int, qk_rope_dim: int, v_head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.qk_nope_dim = qk_nope_dim
        self.qk_rope_dim = qk_rope_dim
        self.v_head_dim = v_head_dim
        
        # Q-compression
        self.q_down = nn.Linear(d_model, q_lora_rank, bias=False)
        self.q_norm = RMSNorm(q_lora_rank)
        self.q_up_nope = nn.Linear(q_lora_rank, n_heads * qk_nope_dim, bias=False)
        self.q_up_rope = nn.Linear(q_lora_rank, n_heads * qk_rope_dim, bias=False)
        
        # KV-compression
        self.kv_down = nn.Linear(d_model, kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(kv_lora_rank)
        self.kv_up_nope = nn.Linear(kv_lora_rank, n_heads * qk_nope_dim, bias=False)
        self.kv_up_rope = nn.Linear(kv_lora_rank, qk_rope_dim, bias=False) # Shared RoPE key
        self.kv_up_v = nn.Linear(kv_lora_rank, n_heads * v_head_dim, bias=False)
        
        self.o_proj = nn.Linear(n_heads * v_head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, kv_cache=None):
        B, T, _ = x.shape
        
        # Q pathway
        c_q = self.q_norm(self.q_down(x))
        q_nope = self.q_up_nope(c_q).view(B, T, self.n_heads, self.qk_nope_dim)
        q_rope = self.q_up_rope(c_q).view(B, T, self.n_heads, self.qk_rope_dim)
        
        # KV pathway
        c_kv = self.kv_norm(self.kv_down(x))
        k_nope = self.kv_up_nope(c_kv).view(B, T, self.n_heads, self.qk_nope_dim)
        k_rope = self.kv_up_rope(c_kv).view(B, T, 1, self.qk_rope_dim) # Shared across heads
        v = self.kv_up_v(c_kv).view(B, T, self.n_heads, self.v_head_dim)
        
        # Apply RoPE (Decoupled)
        q_rope = apply_rotary_emb(q_rope, freqs_cos, freqs_sin)
        k_rope = apply_rotary_emb(k_rope, freqs_cos, freqs_sin)
        
        # Cache handling (caching the latent c_kv is better in practice, but caching expanded k/v for simplicity here)
        if kv_cache is not None:
            k_n_cache, k_r_cache, v_cache = kv_cache
            k_nope = torch.cat([k_n_cache, k_nope], dim=1)
            k_rope = torch.cat([k_r_cache, k_rope], dim=1)
            v = torch.cat([v_cache, v], dim=1)
        kv_cache = (k_nope, k_rope, v)  # always return the cache so the prefill call starts one
        
        # Concatenate nope and rope parts
        q = torch.cat([q_nope, q_rope], dim=-1)
        k_rope_expand = k_rope.expand(-1, -1, self.n_heads, -1)
        k = torch.cat([k_nope, k_rope_expand], dim=-1)
        
        # Transpose for SDPA [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Training / prefill (T>1, empty cache): causal mask. Cached decode step (T=1): attend to all keys.
        is_causal = T > 1
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(attn_out), kv_cache

class DeepSeekMoE(nn.Module):
    def __init__(self, d_model: int, n_routed_experts: int, n_shared_experts: int, expert_ffn_dim: int, top_k: int):
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.top_k = top_k
        
        # Shared expert
        self.shared_expert = MLP(d_model, expert_ffn_dim) if n_shared_experts > 0 else None
        
        # Routed experts
        self.experts = nn.ModuleList([MLP(d_model, expert_ffn_dim) for _ in range(n_routed_experts)])
        self.gate = nn.Linear(d_model, n_routed_experts, bias=False)
        
        # Aux-loss-free balancing bias
        self.register_buffer("expert_bias", torch.zeros(n_routed_experts))
        self.register_buffer("expert_load", torch.zeros(n_routed_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat_x = x.view(-1, D)
        
        # Gate routing
        scores = self.gate(flat_x) # [B*T, num_experts]
        
        # Apply bias for top-k *selection* only
        route_scores = scores + self.expert_bias
        topk_vals, topk_idx = torch.topk(route_scores, self.top_k, dim=-1)
        
        # Weights come from original scores
        topk_weights = torch.gather(scores, 1, topk_idx)
        topk_weights = F.softmax(topk_weights, dim=-1)
        
        # Accumulate per-expert token counts over all micro-batches of an optimizer step;
        # post_step() turns them into the bias update and resets them (no gradients)
        if self.training:
            with torch.no_grad():
                self.expert_load += torch.bincount(topk_idx.flatten(), minlength=self.n_routed_experts).to(self.expert_load.dtype)

        # Dispatch
        flat_out = torch.zeros_like(flat_x)
        for i, expert in enumerate(self.experts):
            idx, k_idx = (topk_idx == i).nonzero(as_tuple=True)
            if idx.numel() > 0:
                tokens = flat_x[idx]
                weights = topk_weights[idx, k_idx].unsqueeze(-1)
                # cast: under autocast the expert output is bf16 while flat_out matches the input dtype
                flat_out.index_add_(0, idx, (expert(tokens) * weights).to(flat_out.dtype))
                
        out = flat_out.view(B, T, D)
        
        # Add shared expert
        if self.shared_expert is not None:
            out = out + self.shared_expert(x)
            
        return out

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, q_lora_rank: int, kv_lora_rank: int, 
                 qk_nope_dim: int, qk_rope_dim: int, v_head_dim: int, 
                 is_dense: bool, n_routed_experts: int, n_shared_experts: int, expert_ffn_dim: int, top_k: int):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadLatentAttention(
            d_model, n_heads, q_lora_rank, kv_lora_rank, qk_nope_dim, qk_rope_dim, v_head_dim
        )
        self.mlp_norm = RMSNorm(d_model)
        
        if is_dense:
            # First layer is traditionally dense in DeepSeek
            self.mlp = MLP(d_model, expert_ffn_dim * 4) 
        else:
            self.mlp = DeepSeekMoE(d_model, n_routed_experts, n_shared_experts, expert_ffn_dim, top_k)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, kv_cache=None):
        h, new_kv = self.attn(self.attn_norm(x), freqs_cos, freqs_sin, kv_cache)
        x = x + h
        x = x + self.mlp(self.mlp_norm(x))
        return x, new_kv

class DeepSeekMLAMoEModel(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int = 768, 
                 n_layers: int = 16, n_heads: int = 12, q_lora_rank: int = 384, 
                 kv_lora_rank: int = 256, qk_nope_dim: int = 64, qk_rope_dim: int = 32, 
                 v_head_dim: int = 64, n_routed_experts: int = 12, n_shared_experts: int = 1, 
                 expert_ffn_dim: int = 384, top_k: int = 2, use_mtp: bool = True):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.use_mtp = use_mtp
        
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            DecoderBlock(
                d_model=d_model, n_heads=n_heads, q_lora_rank=q_lora_rank, kv_lora_rank=kv_lora_rank,
                qk_nope_dim=qk_nope_dim, qk_rope_dim=qk_rope_dim, v_head_dim=v_head_dim,
                is_dense=(i == 0), n_routed_experts=n_routed_experts, n_shared_experts=n_shared_experts,
                expert_ffn_dim=expert_ffn_dim, top_k=top_k
            ) for i in range(n_layers)
        ])
        
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight
        
        # Optional MTP head (predicts t+2)
        if self.use_mtp:
            self.mtp_proj = nn.Sequential(
                nn.Linear(d_model, d_model * 2, bias=False),
                nn.SiLU(),
                nn.Linear(d_model * 2, d_model, bias=False)
            )
        
        freqs_cos, freqs_sin = precompute_freqs_cis(qk_rope_dim, max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.n_layers, self.n_heads = n_layers, n_heads
        self.qk_head_dim, self.v_head_dim = qk_nope_dim + qk_rope_dim, v_head_dim
        self.apply(self._init_weights)
        # scale down residual-branch output projections (attention o_proj and every expert's down proj)
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
        extras = {}
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            extras["ce_loss"] = loss.detach()

            if self.use_mtp and T > 1:
                # Predict t+2 using mtp projection. 
                # logits[t] aims for targets[t] (which is actual token t+1)
                # mtp_logits[t] aims for targets[t+1] (actual token t+2)
                mtp_h = self.mtp_proj(x[:, :-1])
                mtp_logits = self.output(mtp_h)
                
                # We can only predict up to T-2
                mtp_loss = F.cross_entropy(mtp_logits.reshape(-1, mtp_logits.size(-1)), targets[:, 1:].reshape(-1))
                loss = loss + 0.3 * mtp_loss # MTP scaling factor
                extras["mtp_loss"] = mtp_loss.detach()  # no .item(): it forces a GPU sync every micro-batch
                
        return (logits, loss, extras) if extras else (logits, loss)

    def post_step(self, gamma: float = 0.001):
        """Aux-loss-free balancing update."""
        for layer in self.layers:
            if isinstance(layer.mlp, DeepSeekMoE):
                moe = layer.mlp
                if moe.expert_load.sum() == 0:
                    continue
                # Under-loaded experts (below the mean count) get a higher selection bias, over-loaded lower
                moe.expert_bias += gamma * torch.sign(moe.expert_load.mean() - moe.expert_load)
                moe.expert_load.zero_()

    def flops_per_token(self, seq_len: int) -> float:
        """Computes FLOPs based purely on active parameters."""
        active_params = 0
        for name, param in self.named_parameters():
            # Exclude full expert pool, include only active route computation later
            if "experts." not in name:
                active_params += param.numel()
                
        # Add active MoE params per layer
        for layer in self.layers:
            if isinstance(layer.mlp, DeepSeekMoE):
                # We activate 'top_k' experts per token
                expert_params = sum(p.numel() for p in layer.mlp.experts[0].parameters())
                active_params += layer.mlp.top_k * expert_params

        # attention scores (q.k over qk_head_dim) + weighted values (v_head_dim), forward + backward
        attn = 6.0 * self.n_layers * self.n_heads * (self.qk_head_dim + self.v_head_dim) * seq_len
        # the MTP head re-uses the (tied, counted-once) output matrix for a second set of logits
        mtp = 6.0 * self.output.weight.numel() if self.use_mtp else 0.0
        return 6.0 * active_params + attn + mtp

    @torch.no_grad()
    def generate_tokens(self, idx: torch.Tensor, max_new: int, sample_fn) -> torch.Tensor:
        kv_caches = [None] * len(self.layers)
        out_tokens = []
        current_idx = idx
        
        for i in range(max_new):
            B, T = current_idx.shape
            start_pos = 0 if kv_caches[0] is None else kv_caches[0][0].size(1)
            
            freqs_cos = self.freqs_cos[start_pos : start_pos + T]
            freqs_sin = self.freqs_sin[start_pos : start_pos + T]
            
            x = self.tok_embeddings(current_idx)
            new_caches = []
            
            for j, layer in enumerate(self.layers):
                x, new_kv = layer(x, freqs_cos, freqs_sin, kv_caches[j])
                new_caches.append(new_kv)
                
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
    d_model = kw.get("d_model", 768)
    n_layers = kw.get("n_layers", 16)
    n_heads = kw.get("n_heads", 12)
    q_lora_rank = kw.get("q_lora_rank", 384)
    kv_lora_rank = kw.get("kv_lora_rank", 256)
    qk_nope_dim = kw.get("qk_nope_dim", 64)
    qk_rope_dim = kw.get("qk_rope_dim", 32)
    v_head_dim = kw.get("v_head_dim", 64)
    n_routed_experts = kw.get("n_routed_experts", 12)
    n_shared_experts = kw.get("n_shared_experts", 1)
    expert_ffn_dim = kw.get("expert_ffn_dim", 384)
    top_k = kw.get("top_k", 2)
    use_mtp = kw.get("use_mtp", True)
    
    return DeepSeekMLAMoEModel(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_dim=qk_nope_dim,
        qk_rope_dim=qk_rope_dim,
        v_head_dim=v_head_dim,
        n_routed_experts=n_routed_experts,
        n_shared_experts=n_shared_experts,
        expert_ffn_dim=expert_ffn_dim,
        top_k=top_k,
        use_mtp=use_mtp
    )