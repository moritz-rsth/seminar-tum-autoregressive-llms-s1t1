import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModel, AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-135M"  # tiny Llama-architecture LLM, ungated, ~270 MB


def embed(texts):
    """Look up each text in the Llama input-embedding table (576-dim, averaged over its tokens)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    table = AutoModel.from_pretrained(MODEL).get_input_embeddings().weight.float().detach().numpy()
    return np.array([table[tokenizer(" " + t).input_ids].mean(axis=0) for t in texts])


def to_3d(vectors):
    """PCA to 3D. Linear, so arrows from the origin (= the mean vector) keep their meaning."""
    centered = vectors - vectors.mean(axis=0)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    return centered @ components[:3].T


def visualize(points, labels):
    """Rotatable 3D plot: one arrow from the origin to each text's vector."""
    fig = plt.figure("Embedding visualization", figsize=(8, 8))
    fig.suptitle(f"Embedding visualization\n{MODEL} input embeddings, PCA to 3D", fontsize=13)
    ax = fig.add_subplot(projection="3d")
    colors = plt.cm.tab10(np.arange(len(labels)))
    ax.quiver(0, 0, 0, points[:, 0], points[:, 1], points[:, 2], colors=colors, arrow_length_ratio=0.08, linewidth=1.5)
    for (x, y, z), label, color in zip(points, labels, colors):
        ax.text(x, y, z, f" {label}", color=color, fontsize=11)
    ax.scatter(0, 0, 0, color="black", s=15)
    limit = np.abs(points).max()
    for axis, name in enumerate("xyz"):
        end = np.eye(3)[axis] * limit * 1.1
        ax.plot(*np.stack([-end, end]).T, color="gray", linewidth=0.8)
        ax.text(*end, name, color="gray", fontsize=12)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit), xticks=[], yticks=[], zticks=[])
    ax.set_box_aspect(None, zoom=1.4)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    texts = ["Germany", "Austria", "Switzerland", "football", "soccer", "cat", "dog"]

    high_dim_vectors = embed(texts)

    three_dim_vectors = to_3d(high_dim_vectors)

    visualize(three_dim_vectors, texts)
