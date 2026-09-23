from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

N_COMPONENTS = 50
RANDOM_SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", action="store_true",
                        help="run UMAP on the raw 768 dimensions instead of the PCA")
    args = parser.parse_args()

    embeddings = np.load(RESULTS / "embeddings.npy").astype(np.float32)
    print(f"emb {embeddings.shape}", flush=True)

    # Both steps are skipped if their output already exists, so re-running is cheap.
    if not (RESULTS / "pca50.npy").exists():
        pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_SEED).fit(embeddings)
        np.save(RESULTS / "pca50.npy", pca.transform(embeddings).astype(np.float32))
        np.save(RESULTS / "pca_evr.npy", pca.explained_variance_ratio_.astype(np.float32))
        print(
            f"PCA done; top-{N_COMPONENTS} explains "
            f"{pca.explained_variance_ratio_.sum():.1%}",
            flush=True,
        )

    # The two versions are kept side by side rather than overwriting each other, so the
    # cheap PCA-50 map stays available while the expensive one runs.
    output = RESULTS / ("umap_raw768.npy" if args.raw else "umap.npy")
    if output.exists():
        print(f"{output.name} exists, nothing to do", flush=True)
        return

    import umap

    if args.raw:
        coordinates_in = embeddings
        print(f"UMAP on raw 768-d {coordinates_in.shape}", flush=True)
    else:
        coordinates_in = np.load(RESULTS / "pca50.npy").astype(np.float32)
        print(f"UMAP on PCA-50 {coordinates_in.shape}", flush=True)

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
    coordinates = projector.fit_transform(coordinates_in)

    np.save(output, coordinates.astype(np.float32))
    print(f"UMAP done -> results/{output.name}", flush=True)


if __name__ == "__main__":
    main()
