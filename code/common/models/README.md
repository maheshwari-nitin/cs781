# Phase-1 model architectures

**You implement these four files** (course rule §17.5: no LLM-written architecture code).
`_toy_decoder.py` / `_toy_encdec.py` exist only to smoke-test the pipeline. They are not submission models.

| File | Type | Inspiration |
|---|---|---|
| `A_qwen_dense.py` | decoder-only | Qwen3 |
| `B_encdec.py` | encoder-decoder | T5 / UL2 with modern blocks |
| `C_gdn_hybrid.py` | hybrid (linear + softmax attention) | Qwen3-Next |
| `D_mla_moe.py` | efficient (MLA + MoE) | DeepSeek-V2/V3 |

## Interface contract (what `lm_lib` expects)

Every file defines:

```python
MODEL_TYPE = "decoder"            # or "encdec"
def build_model(vocab_size: int, max_seq_len: int, **kw) -> torch.nn.Module: ...
```

`**kw` receives `TrainConfig.extra`, so you can sweep your own config values without editing code.

**Decoder models (A, C, D):**
```
forward(idx: Long[B, T], targets: Long[B, T] | None = None) -> (logits, loss) or (logits, loss, extras)
```
- `logits`: `[B, T, V]` (full sequence; the trainer uses it for the cross-architecture "tail" loss).
- `loss`: mean token cross-entropy, **plus** any auxiliary loss (MoE balance, MTP). `None` if `targets is None`.
- `extras`: optional dict of scalars to log, e.g. `{"aux": ..., "mtp": ...}` (detached tensors, not `.item()`).
  If `loss` includes auxiliary terms, put the plain cross-entropy in `extras["ce_loss"]`.

**Encoder-decoder (B):**
```
forward(src: Long[B, S], dec_in: Long[B, L], targets: Long[B, L] | None = None) -> (logits, loss[, extras])
```
The loader gives `src = window[:768]`, `dec_in = [<|bos|>] + window[768:1023]`, `targets = window[768:1024]`.

**Optional hooks** (used when present):
| Hook | Used by | Purpose |
|---|---|---|
| `flops_per_token(seq_len) -> float` | MFU logging | Implement for MoE (active params) and to include attention FLOPs. Default is 6N |
| `post_step()` | trainer, after each optimizer step | e.g. DeepSeek-V3 aux-loss-free router-bias update |
| `configure_param_groups(weight_decay) -> list[dict]` | optimizer | Custom no-decay sets |
| `generate_tokens(idx, max_new, sample_fn) -> Long[B, n]` | generation (decoder) | KV-cache decoding. `sample_fn(logits[B,V], context[B,*]) -> Long[B]` |
| `encode(src)`, `decode(memory, dec_in) -> logits` | generation (encdec) | Encode the prompt once instead of every step |

**Test before training:** `lm_lib.model_check.check_model("A_qwen_dense", vocab_size)` (the sanity cell in `01_train`)
checks the param budget, shapes, init loss ≈ ln V, **causality** (no future-token leakage), gradient flow,
one-batch overfitting, encoder→decoder wiring and KV-cache vs uncached generation. It also runs on CPU with a
small `extra` config, so you can debug locally before using GPU time.

Rules the trainer enforces: total **unique** parameters (tied embeddings counted once) must be within **50M–300M**.
Aim for ≤ 295M. Tokenizer: 32k byte-level BPE with `<|endoftext|>`=0, `<|pad|>`=1, `<|bos|>`=2.

Note on `torch.compile`: data-dependent shapes (MoE routing) and external Triton kernels (fla) may not compile.
Set `compile=False` in the train config for that run if you hit errors.

---

## A: Qwen3-style dense decoder
Qwen3 Tech Report (arXiv 2505.09388); RoPE (2104.09864); SwiGLU (2002.05202); GQA (2305.13245);
RMSNorm (1910.07467); QK-Norm (2302.05442).

- Pre-norm blocks: `x + Attn(RMSNorm(x))`, `x + MLP(RMSNorm(x))`, final RMSNorm, LM head tied to the embedding.
- Attention: GQA with 16 query heads / 4 KV heads, head_dim 64. **QK-Norm**: RMSNorm on q and k per head, before RoPE.
  RoPE base 10k–1M. Use `F.scaled_dot_product_attention(is_causal=True)` (FlashAttention path). No biases.
- MLP: SwiGLU, `down(silu(gate(x)) * up(x))`.
- Starting size: d_model 1024, 22 layers, ffn 2816 → about 282M params.
  Per layer: attn ≈ 2.62M (q 1024², k/v 2·1024·256, o 1024²), MLP ≈ 8.65M. Embedding 32k·1024 ≈ 33.6M.

## B: Encoder-decoder
T5 (1910.10683); UL2 (2205.05131).

- Encoder: bidirectional self-attention blocks over the prompt (non-causal SDPA).
- Decoder: causal self-attention, then **cross-attention** to the encoder output, then MLP.
- Share one embedding matrix across encoder input, decoder input and LM head.
- Reuse A's components (RMSNorm, RoPE, SwiGLU, GQA) so the comparison isolates the encoder-decoder factorisation.
- Starting size: about 10 encoder + 10 decoder layers at d 1024 → about 286M.
  Decoder layers are about 2.6M bigger (cross-attention).
- Objective: prefix → continuation (directly matches 512 → 128 words). You can add UL2/T5 span corruption as a
  mixed objective later (that would need a loader change: ask).
- Trade-off to discuss in the report: only 25% of window tokens produce a loss, so you get fewer learning signals per FLOP.

## C: Gated DeltaNet hybrid (Qwen3-Next style)
Gated Delta Networks (arXiv 2412.06464); Gated Attention for LLMs (2505.06708); Qwen3-Next blog (Sep 2025).

- Layer pattern: **3 Gated DeltaNet layers : 1 gated softmax-attention layer**, repeated (e.g. 24 layers = 6 × [GDN, GDN, GDN, Attn]).
- Gated DeltaNet token mixer: linear attention with the gated delta rule
  `S_t = α_t · S_{t-1} (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`, `o_t = S_t q_t`.
  Includes short depthwise causal conv on q/k/v, L2-normalised q/k, an output gate and RMSNorm.
  Kernel: `pip install flash-linear-attention`, then either `fla.layers.GatedDeltaNet` (a library block, like
  `nn.MultiheadAttention`) or write the layer yourself around `fla.ops.gated_delta_rule.chunk_gated_delta_rule`.
- Gated attention layers: A's attention with a sigmoid output gate, `o = Attn(x) ⊙ σ(x W_g)`.
- No RoPE needed in GDN layers. Keep RoPE in the attention layers.
- Size: a GDN mixer (q, k, v, gate, o projections + convs) is about 5.3M params/layer, against about 3.7M for gated
  GQA attention. So d 1024 with 24 layers needs ffn about 1920 to stay under 300M (291.6M). ffn 2560 would be 339M.
- fla's constructor argument is `hidden_size` (not `d_model`; unknown kwargs are silently ignored), and each GDN layer
  needs a consecutive `layer_idx` for its recurrent/conv state during cached generation.

## D: DeepSeek-style MLA + MoE
DeepSeek-V2 (2405.04434, MLA); DeepSeekMoE (2401.06066); DeepSeek-V3 (2412.19437, aux-loss-free balancing + MTP);
Wang et al. aux-loss-free balancing (2408.15664).

- **MLA**: compress KV into a latent `c_kv = W_dkv x` (kv_lora_rank about 256). k/v are up-projected from `c_kv`.
  Decoupled RoPE: a small shared rope key (rope dim 32) plus per-head rope query, concatenated with the non-rope
  ("nope") parts (e.g. 64 dims). Optional low-rank q (q_lora_rank about 384).
- **DeepSeekMoE FFN**: 1 shared expert + about 12 fine-grained routed experts (expert ffn about 384), top-2
  sigmoid/softmax routing. Make the first layer dense.
- **Aux-loss-free balancing**: per-expert bias added to the router scores for top-k *selection* only. In `post_step()`,
  bias += γ·sign(mean_load − load_e). Optional tiny sequence-level balance loss.
- Optional **MTP** (multi-token prediction) head: predicts token t+2. It adds a loss term; drop it at inference.
- Starting size: d 768, 16 layers → about 240M total / about 90M active. Report active vs total params and
  implement `flops_per_token` with active params.
- Efficient MoE dispatch: sort tokens by expert, or loop over experts with index_select (fine at this scale).
