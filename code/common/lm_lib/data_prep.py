"""Streaming data preparation for the Phase-1 corpus.

The raw files are single JSON documents of up to ~10.7 GB, so nothing here loads a whole
file into memory: documents are streamed with ijson, sampled, and written to JSONL, then
tokenised into a flat uint16 token file (documents separated by <|endoftext|>).
"""
import json
import os
import random
import time

import numpy as np

SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|bos|>"]
TEXT_KEYS = ("text", "content", "document", "judgment", "judgement", "body", "doc", "data")


def peek(path, n_bytes=1500):
    with open(path, "rb") as f:
        head = f.read(n_bytes)
    print(f"--- {os.path.basename(path)} ({os.path.getsize(path) / 1e9:.2f} GB) ---")
    print(head.decode("utf-8", errors="replace"))
    print("...")


def extract_text(item):
    if isinstance(item, str):
        return item
    if isinstance(item, list):
        parts = [extract_text(x) for x in item]
        return "\n".join(p for p in parts if p)
    if isinstance(item, dict):
        for k in TEXT_KEYS:
            if k in item and item[k]:
                return extract_text(item[k])
        strings = [v for v in item.values() if isinstance(v, str)]
        return max(strings, key=len) if strings else ""
    return ""


def _detect_format(path):
    with open(path, "rb") as f:
        head = f.read(1 << 16)
    stripped = head.lstrip()
    if not stripped:
        raise ValueError(f"{path} looks empty")
    first = chr(stripped[0])
    if first == "[":
        return "array"
    if first == "{":
        # JSONL if the first line is a complete JSON value on its own
        line = stripped.split(b"\n", 1)[0]
        try:
            json.loads(line)
            return "jsonl"
        except json.JSONDecodeError:
            return "object"
    raise ValueError(f"Unrecognised JSON layout in {path}: starts with {first!r}")


def iter_docs(path):
    """Yield document texts from a JSON array, a JSON object (id -> doc) or JSONL file."""
    import ijson

    fmt = _detect_format(path)
    if fmt == "jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    t = extract_text(json.loads(line))
                    if t:
                        yield t
        return
    with open(path, "rb") as f:
        it = ijson.items(f, "item", use_float=True) if fmt == "array" else (
            v for _, v in ijson.kvitems(f, "", use_float=True))
        for item in it:
            t = extract_text(item)
            if t:
                yield t


def sample_sources(files, out_dir, frac=0.10, val_frac=0.005, seed=1337, min_chars=200,
                   normalize_whitespace=False):
    """Document-level random sample from each source.

    files: {source_name: path}. frac: float or {source_name: float} (lets you change mixing).
    normalize_whitespace: collapse all whitespace runs (incl. paragraph breaks) to single spaces,
    which is how the official test prompts are formatted.
    Writes out_dir/train_docs.jsonl and out_dir/val_docs.jsonl; returns per-source stats.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    stats = {}
    with open(os.path.join(out_dir, "train_docs.jsonl"), "w", encoding="utf-8") as ftr, \
         open(os.path.join(out_dir, "val_docs.jsonl"), "w", encoding="utf-8") as fva:
        for name, path in files.items():
            p = frac[name] if isinstance(frac, dict) else frac
            s = dict(seen=0, train_docs=0, val_docs=0, train_chars=0, val_chars=0)
            t0 = time.time()
            for text in iter_docs(path):
                s["seen"] += 1
                r = rng.random()
                if r >= p or len(text) < min_chars:
                    continue
                if normalize_whitespace:
                    text = " ".join(text.split())
                rec = json.dumps({"src": name, "text": text}, ensure_ascii=False)
                if r < p * val_frac:
                    fva.write(rec + "\n"); s["val_docs"] += 1; s["val_chars"] += len(text)
                else:
                    ftr.write(rec + "\n"); s["train_docs"] += 1; s["train_chars"] += len(text)
                if s["seen"] % 200_000 == 0:
                    print(f"  {name}: seen {s['seen']:,} docs ({time.time() - t0:.0f}s)")
            s["seconds"] = round(time.time() - t0, 1)
            stats[name] = s
            print(f"{name}: {s}")
    with open(os.path.join(out_dir, "sample_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def iter_jsonl_texts(path, max_chars=None):
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)["text"]
            yield t
            total += len(t)
            if max_chars and total >= max_chars:
                return


def train_tokenizer(jsonl_path, out_path, vocab_size=32000, max_chars=400_000_000,
                    split_digits=False, min_frequency=2):
    """Byte-level BPE trained on (a prefix of) the sampled corpus."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

    tok = Tokenizer(models.BPE())
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.pre_tokenizer = (pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True), byte_level])
                         if split_digits else byte_level)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    t0 = time.time()
    tok.train_from_iterator(iter_jsonl_texts(jsonl_path, max_chars), trainer=trainer)
    tok.save(out_path)
    print(f"tokenizer: vocab={tok.get_vocab_size()} trained in {time.time() - t0:.0f}s -> {out_path}")
    return tok


def tokenize_to_bin(jsonl_path, bin_path, tokenizer_path, batch_docs=2000):
    """Tokenise every doc, append <|endoftext|>, and write a flat uint16 file."""
    from tokenizers import Tokenizer

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    tok = Tokenizer.from_file(tokenizer_path)
    assert tok.get_vocab_size() < 65536, "uint16 storage needs vocab < 65536"
    eot = tok.token_to_id("<|endoftext|>")
    n_tokens = n_chars = 0
    t0 = time.time()

    def flush(buf, f):
        nonlocal n_tokens
        encs = tok.encode_batch(buf)
        ids = []
        for e in encs:
            ids.extend(e.ids)
            ids.append(eot)
        np.asarray(ids, dtype=np.uint16).tofile(f)
        n_tokens += len(ids)

    with open(bin_path, "wb") as f:
        buf = []
        for text in iter_jsonl_texts(jsonl_path):
            buf.append(text)
            n_chars += len(text)
            if len(buf) >= batch_docs:
                flush(buf, f); buf = []
                print(f"  {n_tokens / 1e6:,.1f}M tokens ({time.time() - t0:.0f}s)", end="\r")
        if buf:
            flush(buf, f)
    print(f"\n{os.path.basename(bin_path)}: {n_tokens:,} tokens, {n_chars / max(n_tokens, 1):.2f} chars/token, "
          f"{time.time() - t0:.0f}s")
    return dict(tokens=n_tokens, chars=n_chars, chars_per_token=n_chars / max(n_tokens, 1))


def tokenize_full_corpus(files, out_dir, tokenizer_path, sample_frac=0.10, val_frac=0.005, seed=1337,
                         min_chars=200, normalize_whitespace=False, batch_docs=2000):
    """Tokenise EVERY document of every source straight from the raw JSON (no intermediate JSONL) into
    out_dir/train_full.bin, holding out exactly the same validation documents as sample_sources(...)
    did with the same (files order, seed, sample_frac, val_frac, min_chars, normalize_whitespace).

    sample_sources draws one r = rng.random() per document (before any filtering) and sends a document
    to validation iff r < sample_frac * val_frac. Replaying the identical random stream here reproduces
    that split, so val.bin / val_prompts.json from the 10% run stay held out and losses stay comparable.
    The held-out docs are also tokenised to out_dir/val_check.bin so the caller can verify the match.
    """
    from tokenizers import Tokenizer

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    tok = Tokenizer.from_file(tokenizer_path)
    assert tok.get_vocab_size() < 65536
    eot = tok.token_to_id("<|endoftext|>")
    rng = random.Random(seed)
    stats, t0 = {}, time.time()

    def flush(buf, f):
        if not buf:
            return 0
        ids = []
        for e in tok.encode_batch(buf):
            ids.extend(e.ids)
            ids.append(eot)
        np.asarray(ids, dtype=np.uint16).tofile(f)
        return len(ids)

    with open(os.path.join(out_dir, "train_full.bin"), "wb") as ftr, \
         open(os.path.join(out_dir, "val_check.bin"), "wb") as fva:
        total = 0
        for name, path in files.items():
            s = dict(docs=0, train_docs=0, val_docs=0, skipped_short=0, train_tokens=0, val_tokens=0)
            p = sample_frac[name] if isinstance(sample_frac, dict) else sample_frac
            val_threshold = p * val_frac
            tbuf, vbuf = [], []
            for text in iter_docs(path):
                s["docs"] += 1
                r = rng.random()                       # same draw order as sample_sources
                if len(text) < min_chars:
                    s["skipped_short"] += 1
                    continue
                if normalize_whitespace:
                    text = " ".join(text.split())
                if r < val_threshold:
                    vbuf.append(text); s["val_docs"] += 1
                else:
                    tbuf.append(text); s["train_docs"] += 1
                if len(tbuf) >= batch_docs:
                    n = flush(tbuf, ftr); s["train_tokens"] += n; total += n; tbuf = []
                    print(f"  {name}: {s['docs']:,} docs | total {total / 1e9:.3f}B tokens ({time.time() - t0:.0f}s)", end="\r")
                if len(vbuf) >= batch_docs:
                    s["val_tokens"] += flush(vbuf, fva); vbuf = []
            n = flush(tbuf, ftr); s["train_tokens"] += n; total += n
            s["val_tokens"] += flush(vbuf, fva)
            stats[name] = s
            print(f"\n{name}: {s}")
    print(f"train_full.bin: {total:,} tokens in {time.time() - t0:.0f}s")
    with open(os.path.join(out_dir, "full_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def make_eval_prompts(val_jsonl, out_dir, n=300, prompt_words=512, cont_words=128, seed=1337):
    """Build a local test set that mimics the official one: 512-word prompt -> 128-word reference."""
    rng = random.Random(seed)
    docs = []
    for line in open(val_jsonl, "r", encoding="utf-8"):
        rec = json.loads(line)
        words = rec["text"].split()
        if len(words) >= prompt_words + cont_words:
            docs.append((rec["src"], words))
    rng.shuffle(docs)
    prompts, refs, srcs = {}, {}, {}
    for i, (src, words) in enumerate(docs[:n], start=1):
        start = rng.randint(0, len(words) - prompt_words - cont_words)
        prompts[str(i)] = {"text": " ".join(words[start:start + prompt_words])}
        refs[str(i)] = {"text": " ".join(words[start + prompt_words:start + prompt_words + cont_words])}
        srcs[str(i)] = src
    for name, obj in [("val_prompts.json", prompts), ("val_refs.json", refs), ("val_srcs.json", srcs)]:
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
    print(f"local eval set: {len(prompts)} prompts (from {len(docs)} eligible val docs)")
    return len(prompts)
