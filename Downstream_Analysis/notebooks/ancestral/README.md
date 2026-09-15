# The 432 reconstructed ancestors

Everything is in [`figure3_ancestral_origin.ipynb`](figure3_ancestral_origin.ipynb), which
builds Figure 3 and Extended Data Figures 5–6 and shows every panel inline. It runs in four
parts:

1. **The three founding nodes** — the common ancestor and the two class ancestors.
2. **Turnover** — how fast parent and child repertoires diverge, and the controls that
   check this is not an artefact of reconstructing on long branches.
3. **The two founding duplications** — what each one gained, and the near-perpendicular
   shift in chemical space, with its null and four controls.
4. **The trajectory** — every node on one time axis, root to present.

`make_main_figure.py` rebuilds the assembled Figure 3 as one vector PDF from `Tables/`.

## Read this before quoting a number

Ancestral repertoire sizes are **relative**. Compare node against node; do not read them as
absolute counts.

The intermediate odorant counts in Part 4 have real variance — one hyper-broad reconstructed
lineage entering the time slice moves them by tens. The endpoints and the shape are solid;
the intermediate integers are not. Part 4 says so where it matters, and shows the blanked
version beside the original.

## Output folders

| Folder | Holds |
|---|---|
| `Figures/` | the panels, as SVG |
| `Tables/` | the numbers behind them, as CSV |
