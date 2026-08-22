"""Step 2 -- turn every molecule into a 768-number vector with MoLFormer-XL.

The descriptor panel from s01 says useful but coarse things about a molecule: how heavy,
how greasy, how many rings. MoLFormer is a language model trained on SMILES strings, and
its embedding captures structure the descriptors miss. We use it for the "are odorants
near each other" question, because nearness there means chemical similarity in a much
richer sense than "similar molecular weight".

** This is the only step that needs a GPU. ** It refuses to start without one rather than
quietly falling back to CPU, which would take days rather than an hour.

Two choices worth knowing about:

  * The model revision is pinned. If it were not, HuggingFace could move the checkpoint
    under us and every number downstream would change with no warning.
  * Everything is float32. Half precision would be faster, but the rounding shuffles which
    molecules count as nearest neighbours, and the neighbour ranking is exactly what the
    kNN result in s04 depends on.

Outputs
-------
results/embeddings.npy    float32, (n_molecules, 768), aligned row-for-row with the index
results/embed_index.csv   inchikey and token length -- the key everything downstream joins on

About an hour for 730k molecules on a T1000; faster on a bigger card.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

MODEL_NAME = "ibm-research/MoLFormer-XL-both-10pct"
MODEL_REVISION = "7b12d946c181a37f6012b9dc3b002275de070314"   # pinned; do not change casually

# The model cannot look at more than this many tokens, so longer molecules are dropped.
MAX_TOKENS = 202
BATCH_SIZE = 256
PROGRESS_EVERY = 200      # report every Nth batch


def main():
    if not torch.cuda.is_available():
        raise SystemExit(
            "no GPU visible.\n"
            "  This is the one step that needs one. Run it on the GPU server;\n"
            "  s01, s03, s04 and s05 are CPU-only and can run anywhere."
        )

    universe = pd.read_parquet(
        RESULTS / "universe.parquet", columns=["inchikey", "smiles", "is_odorant"]
    )
    print(f"universe: {len(universe)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, revision=MODEL_REVISION
    )
    model = AutoModel.from_pretrained(
        MODEL_NAME, deterministic_eval=True, trust_remote_code=True,
        revision=MODEL_REVISION,
    ).cuda().eval()

    # Drop anything too long for the model. Report how many odorants this costs, because
    # losing odorants would matter much more than losing background molecules.
    all_smiles = list(universe.smiles)
    token_counts = np.array([
        len(ids) for ids in tokenizer(all_smiles, padding=False, truncation=False)["input_ids"]
    ])
    short_enough = token_counts <= MAX_TOKENS
    print(
        f"dropping {int((~short_enough).sum())} over {MAX_TOKENS} tokens "
        f"({100 * (~short_enough).mean():.2f}%); odorants dropped: "
        f"{int(universe.is_odorant.values[~short_enough].sum())}",
        flush=True,
    )

    universe = universe[short_enough].reset_index(drop=True)
    token_counts = token_counts[short_enough]
    all_smiles = list(universe.smiles)

    # Process similar-length molecules together. Batches are padded to their longest
    # member, so sorting by length means far less wasted computation -- worth about 3x.
    by_length = np.argsort(token_counts, kind="stable")
    embeddings = np.zeros((len(all_smiles), 768), dtype=np.float32)

    started = time.time()
    n_done = 0
    with torch.no_grad():
        for batch_start in range(0, len(by_length), BATCH_SIZE):
            batch = by_length[batch_start:batch_start + BATCH_SIZE]
            tokens = tokenizer(
                [all_smiles[i] for i in batch],
                padding=True, truncation=True, max_length=MAX_TOKENS,
                return_tensors="pt",
            )
            tokens = {k: v.cuda(non_blocking=True) for k, v in tokens.items()}

            # Write results back to their original positions, undoing the length sort.
            embeddings[batch] = model(**tokens).pooler_output.float().cpu().numpy()
            n_done += len(batch)

            if (batch_start // BATCH_SIZE) % PROGRESS_EVERY == 0:
                elapsed = time.time() - started
                rate = n_done / max(elapsed, 1e-9)
                remaining = (len(by_length) - n_done) / max(rate, 1e-9) / 60
                print(
                    f"  {n_done}/{len(by_length)}  {rate:.0f} mol/s  eta {remaining:.1f}m",
                    flush=True,
                )

    print(f"done in {(time.time() - started) / 60:.1f} min", flush=True)
    assert (np.abs(embeddings).sum(1) == 0).sum() == 0, "all-zero embedding rows"

    np.save(RESULTS / "embeddings.npy", embeddings)
    universe[["inchikey"]].assign(token_len=token_counts).to_csv(
        RESULTS / "embed_index.csv", index=False
    )
    print(f"wrote {RESULTS / 'embeddings.npy'} {embeddings.shape}", flush=True)


if __name__ == "__main__":
    main()
