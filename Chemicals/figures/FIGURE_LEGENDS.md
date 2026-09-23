# Figure legends — chemical space

## Figure 4 | Odorants occupy a plant-derived, volatile corner of natural-product chemical space.

**Whole-figure legend.** Odorants were compared against all non-odorant natural products
in COCONUT (release 08-2026). The two sets are chemically comparable — both are natural
products — so the differences between them are the result rather than a confound, and no
covariate matching was applied. **a**, Odorants occupy a compact, central region of
natural-product chemical space rather than a separate island. **b**, They are drawn
disproportionately from plants and are depleted three- to four-fold in fungal and
bacterial sources. **c**, They come overwhelmingly from fatty-acid metabolism and are
depleted in alkaloid, polyketide and terpenoid pathways. **d**, They are marked by
sulfur- and aldehyde-bearing groups and are almost devoid of nitrogen chemistry. Together
these define odour chemistry as the short-chain, volatile, nitrogen-free output of plant
fatty-acid metabolism.

**a**, Uniform manifold approximation and projection (UMAP) of MoLFormer-XL embeddings for
726,407 molecules: 720,445 non-odorant natural products (sand), 5,208 odorants not in M2OR
(green) and 754 M2OR ligands (purple). Embeddings are the 768-dimensional pooled output of
MoLFormer-XL-both-10pct at a pinned revision, computed in float32. UMAP was run on the full
768 dimensions (`n_neighbors` = 15, `min_dist` = 0.1, cosine metric, random seed 42). Axes
are in arbitrary units and the origin has no meaning; the layout is illustrative only and
every quantitative claim in this figure is made in the full embedding space or on the
molecular descriptors, never from these coordinates. All groups are drawn whole and at the
same point size except M2OR, which is enlarged for visibility.

**b**, Mean proportion of a molecule's recorded source organisms falling in each group, for
odorants (green) and non-odorant natural products (sand). Restricted to the 269,445
molecules with at least one source organism recorded in COCONUT (1,995 odorants; 267,450
others). Organism names were resolved against NCBI taxonomy using a three-pass match (full
name, then genus + species, then genus); 94% resolved. Values beside each pair are fold
change (odorant ÷ other). Bacteria are shown at superkingdom rank, the remaining groups at
kingdom rank. A compositional measure is used rather than presence/absence because odorants
are recorded from 11.7-fold more organisms on average than other natural products (means
50.4 versus 4.3), which would make every taxon appear enriched simultaneously. All four
groups differ significantly (two-sided Mann–Whitney on the per-molecule share,
Benjamini–Hochberg corrected; *q* < 1 × 10⁻¹⁷).

**c**, Biosynthetic pathway assignment (NPClassifier), as a percentage of annotated
molecules: 2,604 odorants and 669,278 non-odorant natural products carry a pathway label.
Six of seven pathways differ significantly (two-sided Fisher's exact test,
Benjamini–Hochberg corrected across the seven tests); Carbohydrates does not
(odds ratio 0.70, 95% confidence interval 0.47–1.09, *q* = 0.10).

**d**, Functional-group enrichment as log₂ odds ratio, odorants (*n* = 5,962) against
non-odorant natural products (*n* = 720,445). Groups were assigned by SMARTS substructure
match. Positive values indicate enrichment in odorants. Fifteen of sixteen groups differ
significantly (two-sided Fisher's exact test, Benjamini–Hochberg corrected); nitrile does
not (odds ratio 1.22, 95% confidence interval 0.95–1.60, *q* = 0.14). Extremes are thiol
(odds ratio 23.9, 95% confidence interval 20.6–27.9) and amide (0.051, 0.041–0.063).

---

## Supplementary Figure | Structural, receptor-class and taxonomic detail of odorant chemical space.

**Whole-figure legend.** Supporting detail for Figure 4. **a**,**b**, Odour space divides
into an aromatic and an aliphatic half that differ in volatility and flexibility rather
than in size. **c**–**f**, The Class I / Class II division of olfactory receptors does not
correspond to any difference in ligand volatility, although the same comparison resolves a
very large difference between all odorants and other natural products. **g**,**h**, At
finer taxonomic ranks the enriched sources are, without exception, culinary plant lineages.

**a**, Odorants on the map of **Fig. 4a**, split by whether they contain a benzene ring:
4,281 aliphatic (72%) and 1,681 aromatic (28%). Non-odorant natural products in pale grey.

**b**, Standardised mean difference between the aromatic and aliphatic halves for seven
descriptors; positive values are higher in aromatics. Aromatic odorants boil higher
(+0.38) and are less flexible (−0.41), while differing little in mass (+0.11) or polar
surface area (+0.11). The fraction of sp³ carbon is deliberately omitted: aromatic carbons
are sp² by definition, so that difference restates the grouping rather than describing it.

**c**–**e**, Position on the map of ligands assigned to receptors preferring Class I only
(*n* = 111), both classes (*n* = 276) or Class II only (*n* = 112). Assignments were
rebuilt from the receptor co-tuning analysis rather than read from file. Non-odorant
natural products in pale grey.

**f**, Estimated normal boiling point (Joback group contribution) by receptor-class
preference, against all other natural products (*n* = 648,119; 275 of the 276 shared
ligands carry an estimate). Each point is one molecule; for the background a random sample
of 800 points is drawn for legibility while all statistics use the full set. Horizontal
bars are medians, vertical bars the interquartile range. The three receptor-class groups
are statistically indistinguishable (Kruskal–Wallis *H* = 1.24, *P* = 0.54; medians 515 K,
529 K and 532 K), whereas all three lie far below other natural products (median 1,092 K;
Cliff's δ = −0.90 for Class I versus other natural products, two-sided Mann–Whitney
*P* < 1 × 10⁻⁶⁰). The comparison therefore has ample power to detect a difference of this
kind, which makes the null result between receptor classes interpretable.

**g**,**h**, Source-organism enrichment at order (**g**) and family (**h**) rank, as log₂
fold change in the mean share of a molecule's source organisms. The five most enriched and
five most depleted taxa are shown, of 45 orders and 42 families that were both significant
(two-sided Mann–Whitney, Benjamini–Hochberg corrected) and accounted for at least 0.5% of
source organisms on one side; the share floor is applied because a ratio between two
near-zero proportions is unstable. Familiar examples are given beneath each taxon name.
Mammalian taxa are excluded throughout: they appear as "sources" because a compound was
measured in milk, breath or body odour rather than synthesised there.

---

## Notes for Methods, not for the legends

Three limitations belong in Methods rather than in a caption:

1. **Joback boiling points are estimates**, fitted mainly on molecules of 2–10 carbons.
   77% of the natural-product background receives a value above its decomposition
   temperature. Only 0.3% of odorants lack an estimate, so the odorant side is sound, but
   absolute volatility statements about the background are not supportable.
2. **COCONUT records where people have looked.** Plants are over-studied for volatiles and
   *Streptomyces* for antibiotics, so the organism comparison is partly between two
   literatures. Both sides are drawn from one database and one curation pipeline, which
   mitigates but does not remove this.
3. **The negative class is not "odorless"** but "not on an odorant list". Unlike Mayhew et
   al. (2022), whose odorless set was verified, the background here certainly contains
   untested odorous compounds.
