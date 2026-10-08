"""
S1-T1 practical, part B: prefill vs token-by-token decoding with Qwen3-0.6B.

Downloads Qwen3-0.6B weights once (~1.5 GB, bf16 safetensors) into $HF_HOME.

Measures
  1. prefill latency / throughput / time-to-first-token for several prompt lengths
  2. KV-cache tensor shapes per layer (8 KV heads vs 16 query heads = GQA)
  3. per-step decode latency WITH KV cache (manual past_key_values loop)
     vs WITHOUT cache (re-run the whole sequence each step)
and saves plots + a JSON with the numbers.

Run:  HF_HOME="$PWD/hf_cache" .venv/bin/python prefill_vs_decode.py [--quick]
"""
import argparse
import json
import platform
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL = "Qwen/Qwen3-0.6B"

p = argparse.ArgumentParser()
p.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
p.add_argument("--quick", action="store_true", help="smaller sweep (~1 min)")
args = p.parse_args()

# ---------------------------------------------------------------- device setup
if args.device == "auto":
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = args.device
if args.dtype == "auto":
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
else:
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]


def sync():
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def timed(fn, repeats=3):
    """Median wall time (s) of fn() with device synchronisation, after one warm-up."""
    fn(); sync()
    ts = []
    for _ in range(repeats):
        sync(); t0 = time.perf_counter()
        out = fn(); sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), out


print(f"torch {torch.__version__} | device={device} | dtype={dtype} | {platform.machine()} {platform.platform()}")
t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(device).eval()
print(f"model loaded in {time.perf_counter()-t0:.1f}s | attn_implementation={model.config._attn_implementation}")
n_params = sum(p.numel() for p in model.parameters())
print(f"parameters (actual, tied embeddings counted once): {n_params/1e6:.1f} M")
cfg = model.config
L, H, KV, hd = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
print(f"config: layers={L} q_heads={H} kv_heads={KV} head_dim={hd} hidden={cfg.hidden_size}")

# A long, natural-language token stream we can slice to any length
base = ("Large language models generate text one token at a time. During prefill the whole prompt "
        "is processed in parallel and the keys and values of every layer are stored in the KV cache. "
        "During decoding each new token attends to all cached keys and values. ")
stream = tok(base * 400, return_tensors="pt")["input_ids"][0]


def prompt(n):
    return stream[:n].unsqueeze(0).to(device)


results = {"device": device, "dtype": str(dtype), "torch": torch.__version__, "model": MODEL}
torch.manual_seed(0)

with torch.inference_mode():
    # ============================================================ 1. PREFILL
    lengths = [16, 64, 256, 512, 1024] if args.quick else [16, 64, 128, 256, 512, 1024, 2048, 4096]
    print("\n=== 1. Prefill (one forward pass over the whole prompt, builds KV cache) ===")
    print(f"{'prompt_len':>10} {'prefill_ms':>11} {'tok/s':>9} {'TTFT_ms':>9} {'ms/token':>9}")
    prefill = []
    for n in lengths:
        ids = prompt(n)

        def run():
            # logits_to_keep=1: only compute the LM head for the last position (what generation needs)
            return model(input_ids=ids, use_cache=True, logits_to_keep=1)

        t, out = timed(run, repeats=3 if n <= 1024 else 2)

        def ttft():  # prefill + picking the first token (greedy)
            o = model(input_ids=ids, use_cache=True, logits_to_keep=1)
            return o.logits[:, -1].argmax(-1).item()

        t_ttft, _ = timed(ttft, repeats=2)
        prefill.append({"n": n, "prefill_s": t, "tok_per_s": n / t, "ttft_s": t_ttft})
        print(f"{n:>10} {t*1e3:>11.1f} {n/t:>9.0f} {t_ttft*1e3:>9.1f} {t*1e3/n:>9.3f}")
    results["prefill"] = prefill

    # ============================================================ 2. KV-cache shapes
    print("\n=== 2. KV cache after prefilling a 100-token prompt ===")
    out = model(input_ids=prompt(100), use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    print(f"cache type: {type(cache).__name__}, layers: {len(cache.layers) if hasattr(cache, 'layers') else len(cache)}")

    def layer_kv(c, i):
        if hasattr(c, "layers"):          # transformers >= 4.56 / 5.x
            return c.layers[i].keys, c.layers[i].values
        return c.key_cache[i], c.value_cache[i]

    k0, v0 = layer_kv(cache, 0)
    for i in (0, 1, L - 1):
        k, v = layer_kv(cache, i)
        print(f"  layer {i:>2}: K {tuple(k.shape)}  V {tuple(v.shape)}  dtype={k.dtype}")
    print(f"  shape = (batch, num_kv_heads={KV}, seq_len, head_dim={hd}); the model has {H} query heads "
          f"-> each KV head is shared by {H//KV} query heads (GQA)")
    total = sum(layer_kv(cache, i)[0].numel() + layer_kv(cache, i)[1].numel() for i in range(L)) * k0.element_size()
    print(f"  total KV bytes for 100 tokens = {total/1024:.0f} KiB -> {total/100/1024:.1f} KiB/token "
          f"(formula 2*L*KV*hd*{k0.element_size()}B = {2*L*KV*hd*k0.element_size()/1024:.1f} KiB)")
    results["kv_shape_layer0"] = list(k0.shape)
    results["kv_bytes_per_token"] = total / 100
    del cache, out

    # ============================================================ 3. DECODE with vs without cache
    P0 = 128 if args.quick else 256
    N = 64 if args.quick else 256
    print(f"\n=== 3. Greedy decoding: prompt {P0} tokens, {N} new tokens ===")
    ids = prompt(P0)

    # ---- (a) WITH KV cache: prefill once, then feed ONE token per step
    def decode_with_cache():
        cache = DynamicCache(config=model.config)
        sync(); t0 = time.perf_counter()
        o = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        nxt = o.logits[:, -1:].argmax(-1)
        sync(); t_prefill = time.perf_counter() - t0
        toks, steps = [nxt.item()], []
        for _ in range(N - 1):
            sync(); t0 = time.perf_counter()
            o = model(input_ids=nxt, past_key_values=o.past_key_values, use_cache=True)
            nxt = o.logits[:, -1:].argmax(-1)
            toks.append(nxt.item())   # .item() also forces a sync
            steps.append(time.perf_counter() - t0)
        return t_prefill, steps, toks

    # ---- (b) WITHOUT cache: re-run the full (growing) sequence every step
    def decode_without_cache():
        seq, steps, toks = ids, [], []
        t_prefill = None
        for s in range(N):
            sync(); t0 = time.perf_counter()
            o = model(input_ids=seq, use_cache=False, logits_to_keep=1)
            nxt = o.logits[:, -1:].argmax(-1)
            toks.append(nxt.item())
            dt = time.perf_counter() - t0
            if s == 0:
                t_prefill = dt
            else:
                steps.append(dt)
            seq = torch.cat([seq, nxt], dim=1)
        return t_prefill, steps, toks

    decode_with_cache()  # warm-up (MPS kernels compile on first use)
    tp_c, st_c, toks_c = decode_with_cache()
    tp_n, st_n, toks_n = decode_without_cache()

    same = toks_c == toks_n
    n_match = next((i for i, (a, b) in enumerate(zip(toks_c, toks_n)) if a != b), len(toks_c))
    print(f"greedy outputs identical with/without cache: {same} (first {n_match}/{N} tokens match)")
    print("  (low-precision kernels for 1-token vs full-sequence shapes can differ in the last bit; small divergence late is benign)")
    print("generated text (with cache):", repr(tok.decode(toks_c)[:200]))

    def summarise(name, steps, total_prefill):
        tot = total_prefill + sum(steps)
        first10 = statistics.mean(steps[:10]) * 1e3
        last10 = statistics.mean(steps[-10:]) * 1e3
        print(f"{name:<14} prefill {total_prefill*1e3:7.1f} ms | step first10 {first10:6.1f} ms  last10 {last10:6.1f} ms | "
              f"median {statistics.median(steps)*1e3:6.1f} ms | decode {len(steps)/sum(steps):6.1f} tok/s | total {tot:5.2f} s")
        return {"prefill_s": total_prefill, "steps_s": steps, "decode_tok_per_s": len(steps) / sum(steps),
                "total_s": tot, "first10_ms": first10, "last10_ms": last10}

    results["decode_with_cache"] = summarise("WITH cache", st_c, tp_c)
    results["decode_without_cache"] = summarise("WITHOUT cache", st_n, tp_n)
    results["greedy_identical"] = same
    results["decode_prompt_len"], results["decode_new_tokens"] = P0, N

    # ---- (c) reference: HF generate() (uses the cache internally)
    gen_ids = prompt(P0)
    model.generate(gen_ids, max_new_tokens=8, do_sample=False)
    sync(); t0 = time.perf_counter()
    g = model.generate(gen_ids, max_new_tokens=N, min_new_tokens=N, do_sample=False)
    sync(); tg = time.perf_counter() - t0
    print(f"model.generate(): {N} tokens in {tg:.2f}s -> {N/tg:.1f} tok/s end-to-end (incl. prefill)")
    results["generate_tok_per_s"] = N / tg

    # ---- memory-bandwidth view of decoding
    wbytes = sum(p.numel() * p.element_size() for p in model.parameters())
    ms = statistics.median(st_c)
    print(f"\nweights = {wbytes/2**30:.2f} GiB; each decode step must stream all of them -> "
          f"effective {wbytes/ms/1e9:.0f} GB/s at {ms*1e3:.1f} ms/step (M1 Pro peak ~200 GB/s)")
    best = max(prefill, key=lambda r: r["tok_per_s"])
    print(f"prefill peak {best['tok_per_s']:.0f} tok/s (n={best['n']}) vs cached decode "
          f"{results['decode_with_cache']['decode_tok_per_s']:.0f} tok/s -> "
          f"{best['tok_per_s']/results['decode_with_cache']['decode_tok_per_s']:.0f}x")
    results["weights_bytes"] = wbytes

json.dump(results, open("results.json", "w"), indent=1)
print("\nsaved results.json")

# ================================================================ plots
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [r["n"] for r in prefill]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ns, [r["prefill_s"] * 1e3 for r in prefill], "o-", color="#1f77b4")
    ax[0].set(xscale="log", yscale="log", xlabel="prompt length (tokens)", ylabel="prefill time (ms)",
              title="Prefill latency (= time to first token)")
    ax[0].grid(True, which="both", alpha=.3)
    ax[1].plot(ns, [r["tok_per_s"] for r in prefill], "o-", color="#2ca02c", label="prefill")
    ax[1].axhline(results["decode_with_cache"]["decode_tok_per_s"], color="#d62728", ls="--",
                  label="decode (KV cache)")
    ax[1].set(xscale="log", yscale="log", xlabel="prompt length (tokens)", ylabel="tokens / second",
              title="Throughput: prefill vs decode")
    ax[1].legend(); ax[1].grid(True, which="both", alpha=.3)
    fig.suptitle(f"{MODEL} on {device} ({str(dtype).split('.')[-1]})")
    fig.tight_layout(); fig.savefig("prefill_vs_length.png", dpi=150)

    from plot_decode_latency import plot as plot_decode_latency
    plot_decode_latency("results.json", "decode_latency.png")
    print("saved prefill_vs_length.png, decode_latency.png")
except ImportError:
    print("matplotlib not installed -> skipping plots")
