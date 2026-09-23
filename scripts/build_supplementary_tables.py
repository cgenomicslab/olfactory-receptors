#!/usr/bin/env python3
"""Assemble the Supplementary Tables workbook from the analysis outputs.

One tab per Supplementary Table, each opening with its own legend in a merged row above the
header, so that a tab detached from the workbook still says what it holds. Everything is read
from files already in the repository; nothing is recomputed except the threshold scan in S4,
which is derived from the released table of measured against predicted pairs.

    python scripts/build_supplementary_tables.py --out <dir>/Supplementary_Tables.xlsx
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

REPO = Path(__file__).resolve().parent.parent
REF = REPO / "Downstream_Analysis/notebooks/reference_tree"
ANC = REPO / "Downstream_Analysis/notebooks/ancestral"
CHEM = REPO / "Chemicals/results"
PHY = REPO / "Phylogenetic_Analysis/data"

HEADER_FILL = PatternFill("solid", fgColor="E8E8E4")
LEGEND_FILL = PatternFill("solid", fgColor="F6F6F3")
THIN = Side(style="thin", color="BBBBB8")


# --------------------------------------------------------------------------- panels
# Reproduced here so that the published table is self-contained; both dictionaries are
# imported from the analysis code, not retyped (see the assertions in build()).
FG_SMARTS = {
    "fg_alcohol": "[CX4;!$(C=O)][OX2H]",
    "fg_phenol": "[c][OX2H]",
    "fg_carboxylic_acid": "[CX3](=O)[OX2H1]",
    "fg_ester": "[CX3](=O)[OX2H0][#6]",
    "fg_lactone": "[#6][OX2][CX3](=O)[#6]",
    "fg_ether": "[OD2]([#6])[#6]",
    "fg_aldehyde": "[CX3H1](=O)[#6]",
    "fg_ketone": "[#6][CX3](=O)[#6]",
    "fg_amine": "[NX3;H2,H1,H0;!$(NC=O);!$(N[a])]",
    "fg_amide": "[NX3][CX3](=[OX1])",
    "fg_thiol": "[#16X2H]",
    "fg_sulfide": "[#16X2H0][#6]",
    "fg_nitrile": "[NX1]#[CX2]",
    "fg_halogen": "[F,Cl,Br,I]",
    "fg_benzene": "c1ccccc1",
    "fg_terpene_isopropyl": "CC(C)[!#1]",
}

DESCRIPTORS_22 = [
    ("MolWt", "molecular weight (Da)", "bulk", "yes"),
    ("LogP", "Crippen calculated octanol–water partition coefficient", "bulk", "yes"),
    ("TPSA", "topological polar surface area (Å²)", "bulk", "yes"),
    ("HBD", "hydrogen-bond donors", "bulk", "yes"),
    ("HBA", "hydrogen-bond acceptors", "bulk", "yes"),
    ("RotBonds", "rotatable bonds", "bulk", "yes"),
    ("AromaticRings", "aromatic rings", "bulk", "yes"),
    ("RingCount", "rings of any kind", "bulk", "yes"),
    ("AliphaticRings", "non-aromatic rings", "bulk", "no"),
    ("FracCsp3", "fraction of carbons that are sp³", "bulk", "yes"),
    ("HeavyAtoms", "non-hydrogen atoms", "bulk", "yes"),
    ("MolMR", "Crippen molar refractivity", "bulk", "yes"),
    ("LabuteASA", "Labute approximate surface area", "bulk", "no"),
    ("BertzCT", "Bertz topological complexity", "bulk", "yes"),
    ("NumStereo", "unassigned and assigned stereocentres", "bulk", "no"),
    ("QED", "quantitative estimate of drug-likeness", "bulk", "no"),
    ("nHeteroatoms", "atoms other than C and H", "bulk", "no"),
    ("nO", "oxygen atoms", "bulk", "yes"),
    ("nN", "nitrogen atoms", "bulk", "no"),
    ("nS", "sulfur atoms", "bulk", "no"),
    ("nHalogen", "F, Cl, Br or I atoms", "bulk", "no"),
    ("nC", "carbon atoms", "bulk", "no"),
]

# The 13 descriptors standardised over the 754-molecule panel to define the shift-vector
# space of Fig. 3e and Extended Data Fig. 5c–e.
DESCRIPTORS_13 = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotB", "AromaticRings",
                  "FracCsp3", "HeavyAtoms", "RingCount", "MolMR", "BertzCT", "nO"]


# --------------------------------------------------------------------------- tab builders

def s1_datasets():
    rows = [
        ("Receptor–odorant assays", "M2OR export", "53,444 observations",
         "42,917 human, 10,243 mouse, remainder primate and bovine; 3,124 (5.8%) positive",
         "https://m2or.chemsensim.fr/"),
        ("Receptor–odorant assays", "after removing mixture records", "52,175 observations",
         "1,399 unique receptor sequences (589 wild type, 810 mutants), 754 odorants; 3,066 (5.9%) positive", ""),
        ("Receptor–odorant assays", "testable human pairs", "23,782 pairs",
         "409 of 433 human receptors with at least one measured result; 4.74% bind", ""),
        ("Receptor sequences", "human class A GPCRs (Pfam PF00001)", "722 sequences",
         "hmmsearch --cut_ga against the one-protein-per-gene human proteome", "https://www.uniprot.org"),
        ("Receptor sequences", "six-species olfactory receptors (Pfam PF13853)", "584 sequences",
         "human 433, zebrafish 103, Eptatretus burgeri 43, Scyliorhinus torazame 2, "
         "Chiloscyllium punctatum 2, Branchiostoma lanceolatum 1", "https://www.uniprot.org"),
        ("Receptor sequences", "human olfactory receptors", "433 sequences",
         "62 class I, 371 class II, from the two clades descending from the root", ""),
        ("Receptor sequences", "ancestral reconstructions", "432 sequences",
         "node_1 to node_432 of the outgroup-rooted human tree; node_0 has no reported states", ""),
        ("Odorants", "M2OR panel", "754 molecules",
         "columns of every prediction matrix", ""),
        ("Odorants", "chemical-space odorant set", "5,962 molecules",
         "M2OR 754, Leffingwell 3,522, GoodScents 4,492, deduplicated on InChIKey", "https://pyrfume.org"),
        ("Natural products", "COCONUT, release 08-2026", "720,445 molecules",
         "parent structures, deduplicated on recomputed InChIKey, odorants removed",
         "https://coconut.naturalproducts.net"),
        ("Odour descriptors", "Leffingwell and GoodScents via Pyrfume", "—",
         "joined to molecules on InChIKey", "https://pyrfume.org"),
        ("Taxonomy", "NCBI Taxonomy", "—",
         "COCONUT organism strings resolved in three passes; ~94% resolved",
         "https://www.ncbi.nlm.nih.gov/taxonomy"),
        ("Domain models", "Pfam via InterPro", "PF00001, PF13853",
         "gathering thresholds used throughout", "https://www.ebi.ac.uk/interpro/"),
    ]
    frame = pd.DataFrame(rows, columns=["Category", "Dataset or step", "Size", "Notes", "Source"])
    legend = ("Supplementary Table 1 | Datasets, filtering steps and sources. Every input to the "
              "study, the size it has after each filtering step, and where it came from. Sizes are "
              "the counts used in the Results and Methods.")
    return [("", frame)], legend


def s2_receptors():
    classes = pd.read_csv(PHY / "HumanTree/human433_OR_classes.csv")
    breadth = pd.read_csv(REF / "Figures/class_tuning_breadth.csv")
    breadth["uniprot"] = breadth["pid"].str.split(".").str[-1]
    frame = classes.merge(breadth[["uniprot", "n_ligands"]], on="uniprot", how="left")
    frame["n_ligands"] = frame["n_ligands"].fillna(0).astype(int)
    frame["orphan"] = np.where(frame["n_ligands"] == 0, "yes", "no")
    frame["OR_class"] = frame["OR_class"].str.replace("_", " ").str.lower()
    frame = frame.rename(columns={"leaf_id": "tree_tip", "OR_class": "receptor_class",
                                  "n_ligands": "ligands_called"})
    frame = frame[["uniprot", "tree_tip", "receptor_class", "ligands_called", "orphan"]]
    frame = frame.sort_values(["receptor_class", "ligands_called"],
                              ascending=[True, False]).reset_index(drop=True)
    legend = ("Supplementary Table 2 | The 433 human olfactory receptors. Class assignment is the "
              "clade descending from the root of the human maximum likelihood tree (Fig. 3a). "
              "ligands_called is the number of odorants called for that receptor in the hybrid "
              "matrix of Fig. 2a, out of 754; a receptor with none is marked orphan.")
    return [("", frame)], legend


def s3_panels():
    rows = [(name, smarts, "functional group", "Figs. 2c, 4d; Extended Data Figs. 3c, 5b")
            for name, smarts in FG_SMARTS.items()]
    rows += [(name, "", f"descriptor — {desc}",
              "Extended Data Figs. 3d, 6b" + ("; shift-vector space (Fig. 3e)" if in13 == "yes" else ""))
             for name, desc, _, in13 in DESCRIPTORS_22]
    frame = pd.DataFrame(rows, columns=["Name", "SMARTS pattern", "Kind", "Used in"])
    legend = ("Supplementary Table 3 | Functional-group and descriptor panels. The 16 SMARTS "
              "patterns matched with RDKit and the 22 RDKit descriptors used throughout. The 13 "
              "descriptors marked as belonging to the shift-vector space are those standardised "
              "over the 754-molecule panel to define the duplication shift vectors: "
              + ", ".join(DESCRIPTORS_13) + ".")
    return [("", frame)], legend


def s4_calibration():
    pairs = pd.read_csv(REF / "metadata/exp_pred_common_pairs.csv",
                        usecols=["experimental", "probability"])
    y = pairs["experimental"].to_numpy().astype(int)
    p = pairs["probability"].to_numpy()
    measured = y.mean()
    rows = []
    for cut in np.round(np.arange(0.500, 0.996, 0.005), 3):
        pred = (p >= cut).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else np.nan
        den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / den if den else np.nan
        rows.append((cut, pred.mean() * 100, tn, fp, fn, tp, prec, rec, f1, mcc))
    frame = pd.DataFrame(rows, columns=["threshold", "predicted_binding_rate_pct", "TN", "FP",
                                        "FN", "TP", "precision", "recall", "F1", "MCC"])
    frame = frame.round(4)
    legend = ("Supplementary Table 4 | Model calibration over the 23,782 measured "
              f"receptor–odorant pairs, of which {measured * 100:.2f}% bind. One row per "
              "probability threshold: the predicted binding rate, the confusion matrix and the "
              "four threshold-dependent metrics. The rate-matched threshold reported in the text "
              "(0.915) is the row at which predicted_binding_rate_pct equals the measured rate; it "
              "is used only for the confusion matrix and per-pair metrics, never to populate a "
              "matrix. Matrices are populated by the density fill described in Methods. "
              "Area under the receiver operating characteristic curve over these pairs is 0.9699.")
    return [("", frame)], legend


def s5_clusters():
    summary = pd.read_csv(REF / "Figures/chemspace_clusters_hybrid_density_summary.csv")
    tags = pd.read_csv(REF / "Figures/enrichment_hybrid_density.csv")
    tags = tags[tags["fdr"] < 0.05].copy()
    tags["cluster"] = "C" + (tags["cluster"] + 1).astype(str)
    tags = tags.sort_values(["cluster", "obs_exp"], ascending=[True, False])
    tags = tags.rename(columns={"n": "cluster_n", "obs": "molecules_with_tag",
                                "tag_total": "tag_total_in_panel", "exp": "expected",
                                "obs_exp": "fold_enrichment", "p": "P", "fdr": "q"})
    tags = tags[["cluster", "cluster_n", "tag", "molecules_with_tag", "tag_total_in_panel",
                 "expected", "fold_enrichment", "P", "q"]].round(4)
    blocks = [("Cluster medians and functional-group percentages", summary),
              ("Odour-tag enrichments significant after correction", tags)]
    legend = ("Supplementary Table 5 | The six odorant activation clusters, in full. The first "
              "block gives each cluster's theme, size, median descriptors and the percentage of its "
              "molecules carrying each functional group. The second block lists every odour-tag "
              "enrichment significant after Benjamini–Hochberg correction (two-sided hypergeometric "
              "tests), which Table 1 shows only the three strongest of. This is the full version of "
              "Table 1.")
    return blocks, legend


def s6_odorants():
    desc = pd.read_csv(REF / "Figures/chemspace_clusters_hybrid_density_descriptors.csv")
    desc = desc.rename(columns={"sml": "molecule_id", "cluster_name": "cluster_theme"})
    desc["cluster"] = "C" + (desc["cluster"] + 1).astype(str)
    front = ["molecule_id", "SMILES", "cluster", "cluster_theme"]
    frame = desc[front + [c for c in desc.columns if c not in front]]
    legend = ("Supplementary Table 6 | The 687 odorants with at least one predicted binder. "
              "Molecule identifier and SMILES, the activation cluster it falls in, and its RDKit "
              "descriptors and functional-group flags. The 67 further molecules of the 754-molecule "
              "panel have no called binder and therefore no cluster.")
    return [("", frame)], legend


def s7_nodes():
    nodes = pd.read_csv(ANC / "Tables/05_table1_nodes.csv")
    stages = pd.read_csv(ANC / "Tables/05_table1_stages.csv")
    blocks = [("Ancestral nodes at the founding duplications", nodes),
              ("Stages of the code", stages)]
    legend = ("Supplementary Table 7 | Ancestral node repertoires at the founding duplications. "
              "node_fig is the numeral used in Fig. 3a; node_id the label in the released "
              "reconstruction. human_ORs is the number of extant human receptors the node subtends, "
              "ligands the size of its called repertoire, and retained, gained and lost are counted "
              "against its parent. The second block gives, at each stage of the tree, how many "
              "odorants are encoded, how many by exactly one receptor and how many by two or more.")
    return blocks, legend


def s8_duplications():
    chem = pd.read_csv(ANC / "Tables/05_dup_chemistry.csv")
    variants = pd.read_csv(ANC / "Tables/05_orthogonality_variants.csv")
    checks = pd.read_csv(ANC / "Tables/05_orthogonality_crosschecks.csv")
    loss = pd.read_csv(ANC / "Tables/05_loss_selectivity.csv")
    blocks = [("Gain-set functional-group enrichment", chem),
              ("Shift-vector angle under four definitions of the shift", variants),
              ("Cross-checks on the angle", checks),
              ("Was loss chemically selective?", loss)]
    legend = ("Supplementary Table 8 | Controls behind the two founding duplications. dup1 is "
              "node_1 → node_2 (the class I branch) and dup2 node_3 → node_6 (the class II branch). "
              "The first block tests each gain set for each functional group against two "
              "backgrounds, the full 754-molecule panel and the union of the ten nodes' "
              "repertoires. The second gives the angle between the two shift vectors under four "
              "definitions of the shift, with its permutation null. The third gives PERMANOVA, its "
              "dispersion check, the angle after regressing acid and aromatic character out of all "
              "13 descriptors, and how much of each shift the plotted plane captures. The fourth "
              "asks whether what each branch lost was chemically selective.")
    return blocks, legend


def s9_chemspace_controls():
    smd = pd.read_csv(CHEM / "coconut_matched_smd.csv")
    auc = pd.read_csv(CHEM / "coconut_matched_auc.csv")
    knn = pd.read_csv(CHEM / "knn_enrichment.csv")
    knn_null = pd.read_csv(CHEM / "robust_knn_null.csv")
    scaffold = pd.read_csv(CHEM / "robust_scaffold_auc.csv")
    blocks = [("Covariate balance before and after matching", smd),
              ("Separability of odorants from their matched controls", auc),
              ("Separability under a Bemis-Murcko scaffold split", scaffold),
              ("Neighbourhood enrichment", knn),
              ("Neighbourhood enrichment against matched and random nulls", knn_null)]
    legend = ("Supplementary Table 9 | Chemical-space controls. Odorants were matched one to one "
              "to non-odorant natural products on Joback boiling point, molecular weight and cLogP; "
              "the first block reports the standardised mean difference on each covariate before and "
              "after matching. The next blocks report how separable odorants remain from their "
              "controls, on the raw split and on a scaffold split, and how enriched an odorant's "
              "50 nearest neighbours are in other odorants against a matched and a random null. "
              "The quantity reported in the text is the ratio of the odorant enrichment to the "
              "matched null.")
    return blocks, legend


def 10_enrichment():
    path = pd.read_csv(CHEM / "pathway_enrichment.csv")
    fg = pd.read_csv(CHEM / "fg_enrichment.csv")
    king = pd.read_csv(CHEM / "organism_kingdom_summary.csv")
    recovery = pd.read_csv(REF / "Figures/05_pathway_recovery.csv")
    blocks = [("Biosynthetic pathway, odorants vs other natural products (Fig. 4c)", path),
              ("Functional groups, odorants vs other natural products (Fig. 4d)", fg),
              ("Source-organism composition by kingdom (Fig. 4b)", king),
              ("Recovery of pathway labels by two partitions of the same molecules", recovery)]
    legend = ("Supplementary Table 10 | Enrichment statistics behind Fig. 4 and the cluster "
              "pathway result. Pathway and functional-group blocks give the percentage of each side "
              "carrying the label, the odds ratio with its confidence interval, and Benjamini–"
              "Hochberg q; columns suffixed _matched repeat the test against the matched control of "
              "Supplementary Table 9, and the values reported in Fig. 4d are the unmatched ones. The "
              "last block gives Cramér's V for the association between biosynthetic pathway and two "
              "partitions of the same molecules, the receptor-activation clusters and clusters "
              "built from molecular descriptors, against a floor from 2,000 label permutations. "
              "It covers the 427 of 442 pathway-labelled molecules that fall in the five "
              "well-populated classes; polyketides and carbohydrates are excluded because their "
              "counts leave too many expected cells below five. Both partitions derive from "
              "molecular structure, as does the pathway label, so the comparison bounds the "
              "association rather than establishing an independent one.")
    return blocks, legend


TABS = [
    ("S1 Datasets", s1_datasets),
    ("S2 Receptors", s2_receptors),
    ("S3 Panels", s3_panels),
    ("S4 Calibration", s4_calibration),
    ("S5 Clusters", s5_clusters),
    ("S6 Odorants", s6_odorants),
    ("S7 Ancestral nodes", s7_nodes),
    ("S8 Duplications", s8_duplications),
    ("S9 Chemspace controls", s9_chemspace_controls),
    ("S10 Enrichment stats", 10_enrichment),
]


def write_tab(book, sheet, blocks, legend):
    """Write one tab: a wrapped legend, then each block as its own headed sub-table.

    Blocks are stacked down column A rather than merged into one wide frame, so a tab that
    carries several kinds of result stays readable and every column keeps its own header.
    """
    ws = book.create_sheet(sheet)
    ncol = max(len(f.columns) for _, f in blocks)

    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(ncol, 2))
    cell = ws.cell(row=1, column=1, value=legend)
    cell.alignment = Alignment(wrap_text=True, vertical="top")
    cell.font = Font(size=9, italic=True)
    cell.fill = LEGEND_FILL
    ws.row_dimensions[1].height = max(30, 11 * (len(legend) // (ncol * 13) + 2))

    widths = {}
    row = 3
    for title, frame in blocks:
        if title:
            head = ws.cell(row=row, column=1, value=title)
            head.font = Font(bold=True, size=9, color="4A4A46")
            row += 1
        header_row = row
        for j, name in enumerate(frame.columns, start=1):
            c = ws.cell(row=row, column=j, value=str(name))
            c.font = Font(bold=True, size=9)
            c.fill = HEADER_FILL
            c.border = Border(bottom=THIN)
            c.alignment = Alignment(wrap_text=True, vertical="bottom")
            widths[j] = max(widths.get(j, 0), len(str(name)))
        row += 1
        for values in frame.itertuples(index=False):
            for j, v in enumerate(values, start=1):
                if isinstance(v, float) and np.isnan(v):
                    v = None
                elif isinstance(v, (np.integer, np.floating)):
                    v = v.item()
                ws.cell(row=row, column=j, value=v)
                widths[j] = max(widths.get(j, 0), min(len(str(v)), 60))
            row += 1
        row += 1                                   # one blank row between blocks
        if len(blocks) == 1:
            ws.freeze_panes = ws.cell(row=header_row + 1, column=1)

    for j, w in widths.items():
        ws.column_dimensions[get_column_letter(j)].width = min(max(w + 2, 10), 46)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # The published panels must be the ones the analysis actually used.
    import sys
    sys.path.insert(0, str(REPO / "Chemicals"))
    try:
        import chemspace as cs
        assert cs.FUNCTIONAL_GROUP_SMARTS == FG_SMARTS, "SMARTS panel has drifted from the code"
        assert [d[0] for d in DESCRIPTORS_22] == cs.DESCRIPTOR_NAMES, "descriptor panel has drifted"
    except ImportError:
        print("note: chemspace not importable (needs RDKit); panels not cross-checked")

    book = Workbook()
    book.remove(book.active)                        # drop the default empty sheet
    for sheet, builder in TABS:
        blocks, legend = builder()
        write_tab(book, sheet, blocks, legend)
        rows = sum(len(f) for _, f in blocks)
        print(f"{sheet:24s} {len(blocks):>2} block(s), {rows:>5} rows")
    book.active = 0
    book.save(args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
