"""
S1-T1 practical, part C: does Mistral's sliding window attention change prefill or decode speed?

Mistral-7B (14.5 GB in bf16) does not fit a 16 GB laptop, so this builds the Mistral architecture
at Qwen3-0.6B size with random weights. Speed does not depend on the weight values, only on the shapes.
The same model runs twice: sliding_window=WINDOW and sliding_window=None (full causal attention).

Measures, for each context length n
  1. prefill latency (one forward pass over n tokens)
  2. decode step latency with the KV cache after an n-token prompt, and how many tokens the cache keeps

Run:  HF_HOME="$PWD/hf_cache" .venv/bin/python sliding_window.py
"""
import json
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import DynamicCache, MistralConfig, MistralForCausalLM

WINDOW = 1024
LENGTHS = [256, 512, 1024, 2048, 4096, 8192]
DECODE_STEPS = 20
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.bfloat16 if device == "mps" else torch.float32


def sync():
    if device == "mps":
        torch.mps.synchronize()


def timed(fn, repeats=3):
    """Median wall time (s) of fn() after one warm-up."""
    fn(); sync()
    ts = []
    for _ in range(repeats):
        sync(); t0 = time.perf_counter()
        fn(); sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def build(window):
    cfg = MistralConfig(vocab_size=32000, hidden_size=1024, intermediate_size=3072, num_hidden_layers=28,
                        num_attention_heads=16, num_key_value_heads=8, head_dim=128,
                        max_position_embeddings=32768, sliding_window=window)
    torch.manual_seed(0)
    return MistralForCausalLM(cfg).to(device, dtype).eval()


@torch.no_grad()
def measure(model, n):
    ids = torch.randint(0, 32000, (1, n), device=device)
    prefill = timed(lambda: model(ids, past_key_values=DynamicCache(config=model.config), use_cache=True))

    cache = DynamicCache(config=model.config)
    out = model(ids, past_key_values=cache, use_cache=True)
    kept = cache.layers[0].keys.shape[-2]
    nxt = out.logits[:, -1:].argmax(-1)
    steps = []
    for _ in range(DECODE_STEPS):
        sync(); t0 = time.perf_counter()
        out = model(nxt, past_key_values=cache, use_cache=True)
        nxt = out.logits[:, -1:].argmax(-1); sync()
        steps.append(time.perf_counter() - t0)
    return prefill * 1e3, statistics.median(steps[2:]) * 1e3, kept


results = {}
for name, window in [(f"sliding window {WINDOW}", WINDOW), ("full attention", None)]:
    model = build(window)
    print(f"\n=== Mistral architecture, {name} | {device} {dtype} | attn={model.config._attn_implementation} ===")
    print(f"{'context':>8} {'prefill_ms':>11} {'ms/token':>9} {'decode_ms':>10} {'KV kept':>8}")
    rows = []
    for n in LENGTHS:
        pre, dec, kept = measure(model, n)
        rows.append(dict(n=n, prefill_ms=pre, decode_ms=dec, kv_kept=kept))
        print(f"{n:>8} {pre:>11.1f} {pre/n:>9.3f} {dec:>10.1f} {kept:>8}")
    results[name] = rows
    del model
    if device == "mps":
        torch.mps.empty_cache()

json.dump(results, open("sliding_window_results.json", "w"), indent=1)

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
for name, rows in results.items():
    ns = [r["n"] for r in rows]
    a.plot(ns, [r["prefill_ms"] for r in rows], "o-", label=name)
    b.plot(ns, [r["decode_ms"] for r in rows], "o-", label=name)
n0, t0 = results["full attention"][0]["n"], results["full attention"][0]["prefill_ms"]
a.plot(LENGTHS, [t0 * n / n0 for n in LENGTHS], "k:", lw=1, label="linear reference")
for ax, title, ylabel in [(a, "Prefill latency", "prefill time (ms)"), (b, "Decode step after an n-token prompt", "ms per new token")]:
    ax.set(xscale="log", title=title, xlabel="context length (tokens)", ylabel=ylabel)
    ax.set_xticks(LENGTHS, [str(n) for n in LENGTHS]); ax.minorticks_off()
    ax.axvline(WINDOW, color="gray", ls="--", lw=1)
    ax.grid(alpha=.3, which="both"); ax.legend()
a.set_yscale("log")
b.set_ylim(bottom=0)
fig.suptitle(f"Mistral architecture at 0.6B size, random weights, {device} {dtype} (dashed line = window {WINDOW})")
fig.tight_layout(); fig.savefig("sliding_window.png", dpi=130)
print("\nsaved sliding_window_results.json, sliding_window.png")
