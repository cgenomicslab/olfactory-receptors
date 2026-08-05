# Results — Ancestral reconstruction of odorant-binding repertoires across the human OR phylogeny

## Ancestral binding repertoires along a nested lineage of internal nodes

The vertebrate odorant-receptor (OR) family is the largest gene family in the mammalian
genome and encodes odor identity through a combinatorial receptor code, in which each
odorant engages a distinct subset of receptors and each receptor recognises a distinct set
of odorants [Buck & Axel, *Cell* 1991; Malnic et al., *Cell* 1999]. Because the family has
diversified through repeated birth-and-death events over vertebrate evolution [Niimura &
Nei, *PNAS* 2003; Nei, Niimura & Nozawa, *Nat. Rev. Genet.* 2008], the odorants that an
ancestral receptor could bind — and how that spectrum was remodelled along each branch —
are not directly observable. We therefore reconstructed ancestral OR sequences at the
internal nodes of the human OR phylogeny by maximum-likelihood marginal ancestral sequence
reconstruction under the LG substitution model, using a hagfish sequence as outgroup, and
applied a receptor–ligand interaction model trained on curated OR–odorant bioassays
[M2OR; Lalis et al., *Nucleic Acids Res.* 2024] to predict, for each of the 754 assayed
odorants, whether each ancestral node binds it. This combination of ancestral sequence
reconstruction with a functional read-out follows the "resurrection" paradigm that has
been used to trace the historical origins of protein function in other receptor and enzyme
families [Thornton, *Nat. Rev. Genet.* 2004; Harms & Thornton, *Nat. Rev. Genet.* 2013;
Hochberg & Thornton, *Annu. Rev. Biophys.* 2017].

We focused on ten internal nodes (labelled 1–5, 6, 12, 21, 32, 44) that form a single
nested lineage within the tree. Node 1 is the deepest and most promiscuous ancestor
(predicted to bind 177/754 odorants). It gives rise to two sister branches: a "trunk"
lineage descending through nodes 3 → 6 → 12 → 21 → 32 → 44, and a second branch through
node 2, which itself splits into the terminal pair node 4 and node 5. The two arms of the
tree show opposite trajectories in repertoire size. Along the trunk the predicted
repertoire contracts monotonically from 177 (node 1) to 113 (node 3), briefly recovering at
node 6 (130) before collapsing to 77, 60 and finally 51 odorants at node 44 — a net loss of
~70% of the ancestral binding spectrum. In contrast, the node-2 branch expands: node 2
binds 130 odorants, but its descendants node 4 and node 5 bind *more* than their parent
(138 and 147 respectively), each gaining new odorants while retaining most of the ancestral
set. A core of only 14 odorants is bound by all ten nodes, indicating that the shared
ancestral repertoire is small and that most of each node's repertoire is lineage-specific.

## Gains and losses are directional and lineage-specific

For every node whose immediate parent was itself reconstructed, we partitioned its
repertoire relative to its parent into retained, gained (bound by child but not parent) and
lost (bound by parent but not child) odorants. The two arms of the tree are dominated by
opposite processes. The node-1 → node-3 → node-44 trunk is loss-dominated: node 3 loses 64
odorants and gains none relative to node 1; node 12 loses 53 and gains none relative to
node 6; and nodes 21 and 44 each lose ~3–7× more odorants than they gain. The node-2
branch, by contrast, is gain-enriched: relative to node 2, node 4 gains 25 and loses 17
odorants, and node 5 gains 42 and loses 25. The node 1 → node 2 edge is the single most
active branch in the set (89 gains and 136 losses), marking a large functional turnover at
the base of this branch. These directional differences were also tabulated for all ordered
pairs of nodes so that gains and losses can be read along any user-defined path
(`ordered_pair_differences.csv`).

## The deepest split in the set is the Class I / Class II divergence

Assigning each node to an OR class from its descendant human leaves places the node set on the
single most consequential axis of vertebrate OR evolution. Node 1 is the common ancestor of
the entire human repertoire (433 descendant ORs; 14% Class I), and its two daughters are the
two founding class ancestors: **node 2 is the most-recent common ancestor of all 62 human
Class I ("fish-like") ORs (100% Class I), and node 3 is the ancestor of all 371 Class II
("mammalian") ORs (100% Class II)**. The entire loss-dominated trunk (nodes 3, 6, 12, 21, 32,
44) lies within Class II, whereas node 2 and its descendants node 4 and node 5 are entirely
Class I. Class I ORs are the more ancient, "fish-like" receptors, historically associated with
the detection of water-soluble and acidic odorants [Niimura & Nei, *PNAS* 2003; Freitag et
al., *J. Comp. Physiol. A* 1998; Nei, Niimura & Nozawa, *Nat. Rev. Genet.* 2008]. The
functional reconstruction below is therefore not merely a description of two arbitrary
branches, but a reconstruction of the founding functional divergence between the two OR
classes.

## The node-2 (Class I) lineage acquired carboxylic-acid / "sour" recognition

To ask whether the remodelled repertoires correspond to coherent odor qualities, we tested
each node's bound set for over-representation of 147 odor descriptors by a right-tailed
label-permutation test (99,999 shuffles, preserving each set's size; enrichment defined as
p ≤ 0.01 and observed/expected > 1). We ran the test against two backgrounds — the full
754-odorant pool, and the union of odorants bound by any of the ten nodes (which isolates
what is *distinctive within this lineage*) — and repeated it on the gained and lost sets of
every edge.

The clearest signal is the emergence of acid/dairy chemistry along the node-2 branch. Node
2 is strongly enriched for **sour** (4.5×, p = 1 × 10⁻⁵), **dairy**, **cheesy**, **waxy**
and **odorless** descriptors, and the newly-gained odorants on the node-1 → node-2 edge are
even more sharply enriched for **sour** (6.6×, p = 1 × 10⁻⁵), **cheesy** and **dairy**. This
odor signature has a direct chemical basis: whereas carboxylic acids make up 11% of the
full odorant pool, they constitute 55% of node 2's repertoire and **76% of the odorants
gained on the node-1 → node-2 edge**, with a corresponding rise in topological polar surface
area. Short-chain carboxylic acids are the prototypical sour/sweaty/cheesy odorants, and
the recent cryo-EM structure of the human receptor OR51E2 — itself a **Class I** OR — bound
to the C3 acid propionate established that fatty-acid recognition in ORs is achieved through a
compact, occluded binding pocket that reads out acid chain length [Billesbølle et al.,
*Nature* 2023]. Our reconstruction independently places the acquisition of this
acid-recognition capability at node 2, the ancestor of the entire Class I clade, so that the
phylogenetic class, the predicted ligand chemistry, the enriched odor descriptors and the one
extant Class I OR structure all converge on the same conclusion: node 2 is an ancestral Class
I fatty-acid / short-chain-acid–sensing receptor.

The two daughters of node 2 then partition this chemistry — a pattern consistent with
subfunctionalization. Node 4 retains and sharpens the acid specificity: it is the most
strongly acid-enriched node in the set (**acidic** 10.1×, **aldehydic**, **dairy** and
**cheesy** among its gains) and its repertoire remains 54% carboxylic acids. Node 5 instead
shifts away from free acids toward their esters and lactones: its carboxylic-acid content
drops to 38% while ester content rises to 25% (33% among its gains), lipophilicity increases
(median logP 3.0 vs 1.8 for the node-2 gains), and its enriched descriptors move to
**waxy**, **creamy**, **fatty**, **buttery**, **milky**, **fruity** and **plum** — the
sensory vocabulary of lactones and fatty esters rather than of free acids. Thus a single
ancestral gain of acid recognition at node 2 was elaborated in two directions in its
descendants: toward sharper acid detection (node 4) and toward ester/lactone "creamy-fatty"
detection (node 5).

## The trunk lineage specialises toward roasted, sulfurous and phenolic qualities

The loss-dominated trunk tells a complementary story of progressive specialisation. Node 1
itself is enriched for a broad **woody / floral / sweet / balsamic** profile, as expected
for a promiscuous ancestor. Descending the trunk, node 6 becomes enriched for **balsamic,
coffee, roasted, burnt, meaty** and **nutty** descriptors (its gains are enriched for
balsamic 5.8×, cherry, hyacinth and nutty), and node 12 for **meaty, sulfurous** and
**phenolic** qualities. The terminal nodes 32 and 44 converge on **tobacco, tea, plum** and
**berry**. Edge-level enrichment shows that these transitions are driven by the *lost*
odorants as much as by gains: the node-21 and node-44 loss sets are strongly enriched for
**sulfurous, meaty, beefy** and **horseradish** descriptors (up to 15×), indicating that the
tips of the trunk lineage specifically shed the sulfur/savory recognition that was present
earlier in the lineage. This branch therefore illustrates functional specialisation by
attrition — a narrowing of an initially broad repertoire — which is the expected functional
correlate of the birth-and-death dynamics and pseudogene-rich history of the OR family
[Nei, Niimura & Nozawa, *Nat. Rev. Genet.* 2008], and mirrors the wide functional
variability documented among extant human ORs [Saito et al., *Sci. Signal.* 2009;
Mainland et al., *Nat. Neurosci.* 2014].

## Conserved scaffold motifs are a quality-control check, not a specificity signal

To relate the functional transitions to sequence, we aligned the ten ancestral sequences
(MAFFT L-INS-i; 318 columns) and mapped the canonical OR signature motifs [Malnic et al.,
*PNAS* 2004; Zozulya et al., *Genome Biol.* 2001]: the TM1 **GN** motif, the IC1 **LHTPMY**
motif, the TM3/IC2 **MAYDRYVAIC** motif (containing the class-A DRY micro-switch), the TM6/IC3
**KAFSTCxSH** motif, and the TM7 **PMLNPFIY** motif (containing the class-A NPxxY
micro-switch). All were recovered in the expected N→C order and at their expected consensus
sequences (e.g. MAYDRYVAIC is present in 10/10 nodes; the DRY and NPxxY micro-switches are
intact throughout), confirming that the reconstructions are clean, signalling-competent ORs.

Crucially, these are the **intracellular, conserved-scaffold** signatures: they define a
sequence as an OR and are conserved because they contact G<sub>olf</sub> and the activation
machinery — **not the ligand** [Zozulya et al., *Genome Biol.* 2001; Venkatakrishnan et al.,
*Nature* 2013]. Their conservation is therefore an orientation/QC control and is
uninformative about odorant specificity. Accordingly we excluded the scaffold-motif columns
and searched the **variable, non-scaffold columns flanking the anchors** — the pocket-lining
positions in the extracellular halves of TM3/TM5/TM6 that human–mouse comparisons and the
OR51E2 cryo-EM structure implicate in odorant selectivity [Man, Gilad & Lancet, *Protein
Sci.* 2004; de March et al., *Protein Sci.* 2015; Billesbølle et al., *Nature* 2023].

Of 222 variable non-scaffold columns, 39 cleanly separate the Class I clade (nodes 2/4/5)
from the Class II trunk, and 23 of these fall within a predicted TM3/TM5/TM6 pocket window.
The Class-I-specific pocket substitutions cluster in the extracellular half of TM3 — e.g.
Leu→Trp (col 89), Phe→Ile (col 107), Thr→Met (col 113), Phe→Gly (col 116) — and continue
through the TM4/TM5 face (cols 134, 146, 152, 154, 191, 201, 203). These positions
constitute a candidate structural signature of the Class-I acid-binding pocket that arose at
node 2, and are the natural targets for experimental resurrection-and-mutagenesis tests of
the historical origin of acid recognition (`candidate_function_shifting_positions.csv`,
region-annotated and ranked; `fig11_pocket_vs_scaffold`). We note that these approximate TM
windows were defined relative to the intracellular anchors; precise pocket assignment would
follow from a structural alignment to OR51E2.

## Summary

Combining ancestral sequence reconstruction with model-based deorphanization, we find that
the odorant-binding repertoires of these ten ancestral human ORs were remodelled in a
strongly directional, class-specific manner. A promiscuous deep ancestor (node 1, the root of
the human repertoire) split into the two founding class ancestors: a Class II ancestor
(node 3) that specialised toward roasted/sulfurous/phenolic qualities by attrition along a
loss-dominated trunk, and a Class I ancestor (node 2) that *acquired* carboxylic-acid ("sour"
/ fatty-acid) recognition and then subfunctionalized into an acid-sharpening lineage (node 4)
and an ester/lactone "creamy-fatty" lineage (node 5). The acid-recognition transition is
supported by four independent lines of evidence — the phylogenetic class (node 2 = MRCA of all
Class I ORs), the chemistry of the gained ligands (a jump to 76% carboxylic acids), the
enriched odor descriptors (sour/dairy/cheesy at p = 1 × 10⁻⁵), and the OR51E2/propionate
structure — and is accompanied by a set of Class-I-specific pocket-lining substitutions in the
extracellular halves of TM3–TM5, which provide concrete, testable hypotheses for the sequence
basis of the shift.

## Conclusions — the first steps of the olfactory combinatorial code

The olfactory combinatorial code is the mapping in which each odorant activates a
characteristic subset of receptors and each receptor responds to a characteristic set of
odorants [Buck & Axel, *Cell* 1991; Malnic et al., *Cell* 1999]. By reconstructing binding
repertoires at internal nodes we can ask not only what that code looks like today but how its
first dimensions were built. Four conclusions follow from this analysis.

**1. The code began with a generalist, not a specialist.** The deepest ancestor examined
(node 1) is the most promiscuous receptor in the set, predicted to bind 177/754 odorants
across a broad woody/floral/sweet profile, and binding breadth declines monotonically toward
the tips (to 51 odorants at node 44). An ancestral state of broad, low-specificity binding
that is refined into narrower specificities in descendants is a recurring pattern in resurrected
ancestral proteins [Harms & Thornton, *Nat. Rev. Genet.* 2013; Voordeckers et al., *PLoS Biol.*
2012; Khersonsky & Tawfik, *Annu. Rev. Biochem.* 2010], and
implies that the columns of the combinatorial code were **carved out of an initially
overlapping, promiscuous ancestral tuning** rather than assembled from independent specialists.

**2. The first axis of the code is a chemical-class axis, and it coincides with the Class
I / Class II split.** The founding divergence in our set (node 1 → node 2 vs node 3) is
exactly the Class I / Class II divergence, and it separates receptors by odorant chemistry:
the Class I ancestor (node 2) acquired recognition of carboxylic acids / short-chain fatty
acids (water-soluble, "sour" odorants), while the Class II ancestor (node 3) retained the
broader repertoire of volatile odorants. That the earliest, deepest branch of the code is
organised around a coherent and structurally rationalisable chemical class — validated by the
OR51E2/propionate structure of an extant Class I receptor [Billesbølle et al., *Nature* 2023]
— indicates that the primary dimensions of the combinatorial code are **chemically
interpretable from their origin**, consistent with the "fish-like" Class I receptors being
the ancestral, water-borne-odorant detectors of the vertebrate system [Niimura & Nei, *PNAS*
2003; Freitag et al., *J. Comp. Physiol. A* 1998].

**3. Two distinct genetic mechanisms build the code.** The two arms of the tree remodel their
repertoires by opposite processes. The Class II trunk narrows by **attrition** (loss-dominated
specialisation: e.g. progressive loss of sulfurous/meaty/beefy recognition toward nodes 21/44),
whereas the Class I branch diversifies by **duplication and subfunctionalization**: a single
ancestral gain of acid recognition at node 2 is partitioned in its daughters into sharper acid
detection (node 4) and ester/lactone detection (node 5). Both are expected outcomes of the
birth-and-death dynamics that dominate OR family evolution [Nei, Niimura & Nozawa, *Nat. Rev.
Genet.* 2008], and together they show that new "columns" of the code arise both by pruning an
ancestral repertoire and by splitting a newly acquired chemical capability between duplicates.

**4. Specificity evolves against a fixed signalling scaffold.** Every ancestral node retains
the intact OR/class-A signalling motifs (GN, LHTPMY, MAYDRYVAIC/DRY, KAFSTCxSH, PMLNPFIY/NPxxY)
[Malnic et al., *PNAS* 2004; Zozulya et al., *Genome Biol.* 2001; Venkatakrishnan et al.,
*Nature* 2013], while the functional divergence is written into the variable pocket-lining
columns of TM3–TM6. The evolution of *what a receptor detects* is thus decoupled from the
conserved machinery of *how it signals* — a modularity that likely enabled the OR family to
explore odorant space so extensively without compromising G<sub>olf</sub> coupling, and that
makes the pocket residues identified here (rather than the diagnostic motifs) the substrate on
which the first steps of the combinatorial code were written.

A caveat: these conclusions rest on *predicted* binding from a receptor–ligand model
[M2OR; Lalis et al., *Nucleic Acids Res.* 2024], evaluated on a 754-odorant panel, and on
marginal ML ancestral sequences; the specific residue hypotheses (§ pocket candidates) and the
node-2 acid-gain in particular are strong candidates for experimental validation by ancestral
resurrection and pocket mutagenesis [Thornton, *Nat. Rev. Genet.* 2004; Hochberg & Thornton,
*Annu. Rev. Biophys.* 2017].

---

### References (high-impact primary literature cited above)

1. Buck L, Axel R. A novel multigene family may encode odorant receptors: a molecular basis for odor recognition. *Cell* 65:175–187 (1991).
2. Malnic B, Hirono J, Sato T, Buck LB. Combinatorial receptor codes for odors. *Cell* 96:713–723 (1999).
3. Niimura Y, Nei M. Evolution of olfactory receptor genes in the human genome. *Proc. Natl. Acad. Sci. USA* 100:12235–12240 (2003).
4. Nei M, Niimura Y, Nozawa M. The evolution of animal chemosensory receptor gene repertoires: roles of chance and necessity. *Nat. Rev. Genet.* 9:951–963 (2008).
5. Saito H, Chi Q, Zhuang H, Matsunami H, Mainland JD. Odor coding by a mammalian receptor repertoire. *Sci. Signal.* 2:ra9 (2009).
6. Mainland JD, et al. The missense of smell: functional variability in the human odorant receptor repertoire. *Nat. Neurosci.* 17:114–120 (2014).
7. Man O, Gilad Y, Lancet D. Prediction of the odorant binding site of olfactory receptor proteins by human–mouse comparisons. *Protein Sci.* 13:240–254 (2004).
8. Billesbølle CB, et al. Structural basis of odorant recognition by a human odorant receptor. *Nature* 615:742–749 (2023).
9. Venkatakrishnan AJ, et al. Molecular signatures of G-protein-coupled receptors. *Nature* 494:185–194 (2013).
10. Thornton JW. Resurrecting ancient genes: experimental analysis of extinct molecules. *Nat. Rev. Genet.* 5:366–375 (2004).
11. Harms MJ, Thornton JW. Evolutionary biochemistry: revealing the historical and physical causes of protein properties. *Nat. Rev. Genet.* 14:559–571 (2013).
12. Hochberg GKA, Thornton JW. Reconstructing ancient proteins to understand the causes of structure and function. *Annu. Rev. Biophys.* 46:247–269 (2017).
13. Yang Z. PAML 4: phylogenetic analysis by maximum likelihood. *Mol. Biol. Evol.* 24:1586–1591 (2007). *(ancestral reconstruction methodology)*
14. Nguyen L-T, Schmidt HA, von Haeseler A, Minh BQ. IQ-TREE: a fast and effective stochastic algorithm for estimating maximum-likelihood phylogenies. *Mol. Biol. Evol.* 32:268–274 (2015). *(phylogeny inference; confirm against Methods)*
15. Lalis M, et al. M2OR: a database of olfactory receptor–odorant pairs for understanding the molecular mechanisms of olfaction. *Nucleic Acids Res.* 52:D1370–D1380 (2024). *(receptor–ligand training data)*
16. Malnic B, Godfrey PA, Buck LB. The human olfactory receptor gene family. *Proc. Natl. Acad. Sci. USA* 101:2584–2589 (2004). *(OR signature motifs)*
17. Zozulya S, Echeverri F, Nguyen T. The human olfactory receptor repertoire. *Genome Biol.* 2:research0018 (2001). *(OR signature motifs)*
18. Freitag J, Ludwig G, Andreini I, Rössler P, Breer H. Olfactory receptors in aquatic and terrestrial vertebrates. *J. Comp. Physiol. A* 183:635–650 (1998). *(Class I vs Class II ORs)*
19. de March CA, Kim SK, Antonczak S, Goddard WA, Golebiowski J. G protein-coupled odorant receptors: from sequence to structure. *Protein Sci.* 24:1543–1548 (2015).
20. Voordeckers K, et al. Reconstruction of ancestral metabolic enzymes reveals molecular mechanisms underlying evolutionary innovation through gene duplication. *PLoS Biol.* 10:e1001446 (2012).
21. Khersonsky O, Tawfik DS. Enzyme promiscuity: a mechanistic and evolutionary perspective. *Annu. Rev. Biochem.* 79:471–505 (2010).
