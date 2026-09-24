"""UMAP of the MoLFormer embedding, used for plotting only.

Output: results/umap.npy
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

RANDOM_SEED = 42


def main():
    output = RESULTS / "umap.npy"
    if output.exists():
        print(f"{output.name} exists, nothing to do", flush=True)
        return

    import umap

    embeddings = np.load(RESULTS / "embeddings.npy").astype(np.float32)
    print(f"UMAP on the raw embedding {embeddings.shape}", flush=True)

    # Fixing random_state makes the layout reproducible, at the cost of running on one
    # core -- UMAP disables its parallelism when it is given a seed.
    projector = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=RANDOM_SEED,
        verbose=True,
        low_memory=False,
    )
    coordinates = projector.fit_transform(embeddings)

    np.save(output, coordinates.astype(np.float32))
    print(f"UMAP done -> results/{output.name}", flush=True)


if __name__ == "__main__":
    main()
