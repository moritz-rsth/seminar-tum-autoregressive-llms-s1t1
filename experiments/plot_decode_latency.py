"""
Redraw decode_latency.png from results.json (no model needed): the prompt region, the prefill pass
as ms per prompt token, then the ms per generated token of every decode step.

Run:  .venv/bin/python plot_decode_latency.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RED, BLUE = "#d62728", "#1f77b4"


def plot(results_path="results.json", out="decode_latency.png"):
    r = json.load(open(results_path))
    p0, n = r["decode_prompt_len"], r["decode_new_tokens"]
    runs = [("without KV cache (recompute all)", r["decode_without_cache"], RED),
            ("with KV cache (1 new token)", r["decode_with_cache"], BLUE)]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axvspan(0, p0, color="0.94", zorder=0)
    ax.text(p0 / 2, 0.97, f"prompt: {p0} tokens · one prefill pass", transform=ax.get_xaxis_transform(),
            ha="center", va="top", color="0.35", fontsize=10)

    x = list(range(p0 + 1, p0 + n))
    for label, run, color in runs:
        ax.plot(x, [s * 1e3 for s in run["steps_s"]], color=color, lw=1, label=label)
    pre = r["decode_with_cache"]["prefill_s"] * 1e3 / p0
    ax.plot([0, p0], [pre, pre], color="0.15", lw=4, solid_capstyle="butt", zorder=3, label="prefill (same with or without cache)")
    dec = sorted(r["decode_with_cache"]["steps_s"])[len(r["decode_with_cache"]["steps_s"]) // 2] * 1e3
    ax.annotate(f"prefill: {pre:.2f} ms per token", xy=(p0 / 2, pre), xytext=(p0 / 2, 75), ha="center", fontsize=11,
                fontweight="bold", arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.annotate(f"decode: {dec:.0f} ms per token", xy=(p0 + n * 0.55, dec), xytext=(p0 + n * 0.55, 105), ha="center",
                fontsize=11, fontweight="bold", color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE))

    ax.set(xlim=(0, p0 + n), xlabel="token position", ylabel="ms per token",
           title=f"{r['model']} on {r['device']} · {p0}-token prompt, then {n} new tokens")
    ax.set_ylim(bottom=-8)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), fontsize=9, frameon=False)
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out, dpi=150)


if __name__ == "__main__":
    plot()
    print("saved decode_latency.png")
