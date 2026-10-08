"""
S1-T1 practical, part A: inspect dense decoder-only checkpoint configurations.

Downloads ONLY config.json (a few KB each) for several models, plus the
tokenizer for Qwen3-0.6B (~11 MB). No weights are downloaded here.

Run:  HF_HOME="$PWD/hf_cache" .venv/bin/python inspect_config.py
"""
import json
import os

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-8B",
    "mistralai/Mistral-7B-v0.1",   # may be gated -> skipped gracefully
    "meta-llama/Llama-3.2-1B",     # gated: needs HF token + license acceptance
]

FIELDS = [
    ("model_type", "model_type"),
    ("num_hidden_layers", "layers"),
    ("hidden_size", "hidden (d_model)"),
    ("intermediate_size", "FFN intermediate"),
    ("num_attention_heads", "query heads"),
    ("num_key_value_heads", "KV heads"),
    ("head_dim", "head_dim"),
    ("vocab_size", "vocab"),
    ("max_position_embeddings", "max positions"),
    ("rope_theta", "rope_theta"),
    ("rms_norm_eps", "rms_norm_eps"),
    ("tie_word_embeddings", "tie embeddings"),
    ("hidden_act", "activation"),
    ("sliding_window", "sliding_window"),
    ("use_sliding_window", "use_sliding_window"),
    ("attention_bias", "attention_bias"),
    ("torch_dtype", "stored dtype"),
]


def load_config(repo):
    try:
        path = hf_hub_download(repo, "config.json")
    except Exception as e:  # gated / missing
        msg = str(e).splitlines()[0][:110]
        print(f"  [skip] {repo}: {type(e).__name__}: {msg}")
        return None
    with open(path) as f:
        cfg = json.load(f)
    cfg.setdefault("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    cfg.setdefault("num_key_value_heads", cfg["num_attention_heads"])
    if "rope_theta" not in cfg and isinstance(cfg.get("rope_parameters"), dict):
        cfg["rope_theta"] = cfg["rope_parameters"].get("rope_theta")
    if "torch_dtype" not in cfg and "dtype" in cfg:
        cfg["torch_dtype"] = cfg["dtype"]
    return cfg


def param_breakdown(c):
    """Analytic parameter count for a Llama/Qwen3-style dense decoder."""
    d, L, V = c["hidden_size"], c["num_hidden_layers"], c["vocab_size"]
    H, KV, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    ff = c["intermediate_size"]
    bias = c.get("attention_bias", False)

    attn = d * H * hd + 2 * d * KV * hd + H * hd * d          # q, k, v, o
    if bias:
        attn += H * hd + 2 * KV * hd
    qk_norm = 2 * hd if c["model_type"] == "qwen3" else 0     # Qwen3 QK-norm
    mlp = 3 * d * ff                                          # gate, up, down (SwiGLU)
    norms = 2 * d                                             # input + post-attn RMSNorm

    emb = V * d
    lm_head = 0 if c.get("tie_word_embeddings", False) else V * d
    out = {
        "embedding": emb,
        "lm_head (untied)": lm_head,
        "attention": L * (attn + qk_norm),
        "MLP": L * mlp,
        "norms": L * norms + d,
    }
    out["TOTAL"] = sum(out.values())
    out["non-embedding"] = out["TOTAL"] - emb - lm_head
    return out


def kv_bytes_per_token(c, bytes_per_elem=2):
    # K and V, per layer, per KV head, head_dim elements
    return 2 * c["num_hidden_layers"] * c["num_key_value_heads"] * c["head_dim"] * bytes_per_elem


def fmt_params(n):
    return f"{n/1e9:6.3f} B" if n >= 1e9 else f"{n/1e6:7.1f} M"


def main():
    print(f"HF_HOME = {os.environ.get('HF_HOME', '(default ~/.cache/huggingface)')}\n")
    print("Fetching config.json files ...")
    cfgs = {}
    for repo in MODELS:
        c = load_config(repo)
        if c:
            cfgs[repo] = c
    names = list(cfgs)
    short = [n.split("/")[-1] for n in names]

    # ---- 1. side-by-side configuration table
    w = 22
    print("\n=== 1. Configuration comparison ===")
    print(f"{'field':<20}" + "".join(f"{s:>{w}}" for s in short))
    for key, label in FIELDS:
        row = f"{label:<20}"
        for n in names:
            row += f"{str(cfgs[n].get(key, '-')):>{w}}"
        print(row)
    row = f"{'GQA ratio (Q/KV)':<20}"
    for n in names:
        c = cfgs[n]
        row += f"{c['num_attention_heads'] // c['num_key_value_heads']:>{w}}"
    print(row)
    row = f"{'Q-proj width H*hd':<20}"
    for n in names:
        c = cfgs[n]
        row += f"{c['num_attention_heads'] * c['head_dim']:>{w}}"
    print(row)

    # ---- 2. parameter breakdown
    print("\n=== 2. Parameter breakdown (analytic, from config) ===")
    bds = {n: param_breakdown(cfgs[n]) for n in names}
    print(f"{'component':<20}" + "".join(f"{s:>{w}}" for s in short))
    for k in bds[names[0]]:
        row = f"{k:<20}"
        for n in names:
            v = bds[n][k]
            pct = 100 * v / bds[n]["TOTAL"]
            row += f"{fmt_params(v) + f' ({pct:4.1f}%)':>{w}}"
        print(row)
    row = f"{'bf16 weight size':<20}"
    for n in names:
        row += f"{bds[n]['TOTAL'] * 2 / 2**30:>{w-4}.2f} GiB"
    print(row)

    # ---- 3. KV cache
    print("\n=== 3. KV-cache size (bf16/fp16 = 2 bytes/elem, batch 1) ===")
    print(f"{'context':<20}" + "".join(f"{s:>{w}}" for s in short))
    row = f"{'per token':<20}"
    for n in names:
        row += f"{kv_bytes_per_token(cfgs[n]) / 1024:>{w-4}.1f} KiB"
    print(row)
    for T in (2048, 8192, 32768):
        row = f"{f'{T} tokens':<20}"
        for n in names:
            row += f"{kv_bytes_per_token(cfgs[n]) * T / 2**30:>{w-4}.3f} GiB"
        print(row)
    row = f"{'(if MHA, 32K)':<20}"
    for n in names:
        c = dict(cfgs[n]); c["num_key_value_heads"] = c["num_attention_heads"]
        row += f"{kv_bytes_per_token(c) * 32768 / 2**30:>{w-4}.3f} GiB"
    print(row)
    print("Formula: bytes/token = 2 (K,V) x layers x kv_heads x head_dim x bytes_per_elem")

    # ---- 4. tokenization
    print("\n=== 4. Tokenization example (Qwen3 tokenizer) ===")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    s = "The KV cache makes autoregressive decoding at TUM München fast."
    ids = tok(s)["input_ids"]
    print(f"text   : {s!r}")
    print(f"#tokens: {len(ids)}  (chars: {len(s)})")
    print(f"ids    : {ids}")
    print(f"pieces : {[tok.decode([i]) for i in ids]}")
    print(f"tokenizer vocab (len(tok)) = {len(tok)}  vs  config vocab_size = "
          f"{cfgs['Qwen/Qwen3-0.6B']['vocab_size']} (embedding matrix is padded)")
    msgs = [{"role": "user", "content": "What is a KV cache?"}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print("chat template renders as:\n" + chat)


if __name__ == "__main__":
    main()
