#!/usr/bin/env python3
"""Assemble the Supplementary Tables workbook from the analysis outputs.

One tab per Supplementary Table. The first row of each tab holds the table's title, or its full
legend when a legends file is given. Everything is read from files already in the repository;
nothing is recomputed except the threshold scan in Table 11, which is derived from the table of
measured against predicted pairs.

    python Supplementary_Material/supplementary_tables/build_supplementary_tables.py \
        --out Supplementary_Material/supplementary_tables/Supplementary_Tables.xlsx [--legends legends.md]

The legends file is plain text with one paragraph per table, each paragraph starting with
"Supplementary Table N |". Tables without a paragraph keep their title.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

REPO = Path(__file__).resolve().parents[2]
REF = REPO / "Downstream_Analysis/extant_receptors/results"
ANC = REPO / "Downstream_Analysis/ancestral_receptors/results"
CHEM = REPO / "Chemicals/results"
SUPP = REPO / "Supplementary_Material"
PHY = REPO / "Phylogenetic_Analysis/data"

HEADER_FILL = PatternFill("solid", fgColor="E8E8E4")
LEGEND_FILL = PatternFill("solid", fgColor="F6F6F3")
THIN = Side(style="thin", color="BBBBB8")


# --------------------------------------------------------------------------- panels
import sys  # noqa: E402
sys.path.insert(0, str(REPO / "scripts"))
from functional_groups import FUNCTIONAL_GROUPS as FG_SMARTS  # noqa: E402

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

# --------------------------------------------------------------------------- tab builders

def datasets():
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
    title = "Supplementary Table 9 | Datasets, filtering steps and sources"
    return [("", frame)], title


def receptors():
    classes = pd.read_csv(PHY / "HumanTree/human433_OR_classes.csv")
    breadth = pd.read_csv(REF / "class_breadth.csv")
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
    title = "Supplementary Table 3 | The 433 human olfactory receptors"
    return [("", frame)], title


def panels():
    rows = [(name, smarts, "functional group", "Figs. 2c, 3e, 4d; Extended Data Figs. 4b, 6b")
            for name, smarts in FG_SMARTS.items()]
    rows += [(name, "", f"descriptor — {desc}",
              "Extended Data Figs. 3d, 6b" + ("; shift-vector space (Fig. 3e)" if in13 == "yes" else ""))
             for name, desc, _, in13 in DESCRIPTORS_22]
    frame = pd.DataFrame(rows, columns=["Name", "SMARTS pattern", "Kind", "Used in"])
    title = "Supplementary Table 13 | Functional-group and descriptor panels"
    return [("", frame)], title


def calibration():
    pairs = pd.read_csv(REF / "model_check_pairs.csv",
                        usecols=["experimental", "probability"])
    y = pairs["experimental"].to_numpy().astype(int)
    p = pairs["probability"].to_numpy()
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
    title = "Supplementary Table 11 | Model calibration over the 23,782 measured receptor–odorant pairs"
    return [("", frame)], title


def cluster_numbers():
    """PAM label -> "C1".."C6", the cluster numbering of Table 1 and the figures."""
    clusters = pd.read_csv(REF / "odorant_clusters.csv")
    return dict(zip(clusters["cluster"], clusters["cluster_tag"]))


def clusters():
    summary = pd.read_csv(REF / "cluster_summary.csv")
    tags = pd.read_csv(REF / "cluster_tag_enrichment.csv")
    tags = tags[tags["fdr"] < 0.05].copy()
    tags["cluster"] = tags["cluster"].map(cluster_numbers())
    tags = tags.sort_values(["cluster", "obs_exp"], ascending=[True, False])
    tags = tags.rename(columns={"n": "cluster_n", "obs": "molecules_with_tag",
                                "tag_total": "tag_total_in_panel", "exp": "expected",
                                "obs_exp": "fold_enrichment", "p": "P", "fdr": "q"})
    tags = tags[["cluster", "cluster_n", "tag", "molecules_with_tag", "tag_total_in_panel",
                 "expected", "fold_enrichment", "P", "q"]].round(4)
    blocks = [("Cluster medians and functional-group percentages", summary),
              ("Odour-tag enrichments significant after correction", tags)]
    title = "Supplementary Table 4 | The six odorant activation clusters, in full"
    return blocks, title


def odorants():
    desc = pd.read_csv(REF / "cluster_odorants.csv")
    themes = pd.read_csv(REF / "odorant_clusters.csv").drop_duplicates("cluster")
    desc["cluster_theme"] = desc["cluster"].map(dict(zip(themes["cluster"], themes["theme"])))
    desc["cluster"] = desc["cluster_tag"]
    desc = desc.drop(columns="cluster_tag").rename(columns={"sml": "molecule_id"})
    front = ["molecule_id", "SMILES", "cluster", "cluster_theme"]
    frame = desc[front + [c for c in desc.columns if c not in front]]
    title = "Supplementary Table 12 | The 687 odorants with at least one predicted binder"
    return [("", frame)], title


def nodes():
    nodes = pd.read_csv(ANC / "table1_nodes.csv")
    stages = pd.read_csv(ANC / "table1_stages.csv")
    blocks = [("Ancestral nodes at the founding duplications", nodes),
              ("Stages of the code", stages)]
    title = "Supplementary Table 7 | Ancestral node repertoires at the founding duplications"
    return blocks, title


def duplications():
    chem = pd.read_csv(ANC / "dup_chemistry.csv")
    variants = pd.read_csv(ANC / "orthogonality_variants.csv")
    checks = pd.read_csv(ANC / "orthogonality_crosschecks.csv")
    loss = pd.read_csv(ANC / "loss_selectivity.csv")
    blocks = [("Gain-set functional-group enrichment", chem),
              ("Shift-vector angle under four definitions of the shift", variants),
              ("Cross-checks on the angle", checks),
              ("Was loss chemically selective?", loss)]
    title = "Supplementary Table 8 | Controls behind the two founding duplications"
    return blocks, title


def chemspace_controls():
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
    title = "Supplementary Table 14 | Chemical-space controls"
    return blocks, title


def enrichment():
    path = pd.read_csv(CHEM / "pathway_enrichment.csv")
    fg = pd.read_csv(CHEM / "fg_enrichment.csv")
    fg_seeds = pd.read_csv(CHEM / "robust_fg_matched.csv")
    # The compositional test (c07): mean share of a molecule's sources per kingdom,
    # with bacteria taken at domain rank, as drawn in Fig. 4b.
    ranks = pd.read_csv(CHEM / "organism_rank_enrichment.csv")
    king = pd.concat([
        ranks[(ranks["rank"] == "kingdom")
              & ranks.taxon.isin(["Viridiplantae", "Metazoa", "Fungi"])],
        ranks[(ranks["rank"] == "domain") & (ranks.taxon == "Bacteria")],
    ])
    recovery = pd.read_csv(REF / "pathway_association.csv")
    blocks = [("Biosynthetic pathway, odorants vs other natural products (Fig. 4c)", path),
              ("Functional groups, odorants vs other natural products (Fig. 4d)", fg),
              ("Matched functional-group odds ratios under five matching seeds", fg_seeds),
              ("Source-organism composition by kingdom (Fig. 4b)", king),
              ("Recovery of pathway labels by two partitions of the same molecules", recovery)]
    title = "Supplementary Table 6 | Enrichment statistics behind Fig. 4 and the cluster pathway result"
    return blocks, title


def model_comparison():
    summary = pd.read_csv(SUPP / "model_comparison/Tables/model_comparison_summary.csv")
    held_out = pd.read_csv(SUPP / "model_runs/Tables/held_out_test_metrics.csv")
    earlier = pd.read_csv(SUPP / "model_comparison/Tables/earlier_attempts.csv").drop(columns="source")
    blocks = [("The binding model on the held-out test pairs of each training run", held_out),
              ("Baselines on random, unseen-receptor and unseen-odorant splits", summary),
              ("Earlier models trained on M2OR", earlier)]
    title = "Supplementary Table 1 | Performance of the binding model and of other models"
    return blocks, title


def calling_rule_and_runs():
    rules = SUPP / "calling_rule/Tables"
    runs = SUPP / "model_runs/Tables"
    blocks = [("Extant results under each calling rule", pd.read_csv(rules / "extant_by_calling_rule.csv")),
              ("Ancestral results under each calling rule", pd.read_csv(rules / "ancestral_by_calling_rule.csv")),
              ("Barcode and odour-descriptor distance under each calling rule",
               pd.read_csv(rules / "perception_by_calling_rule.csv")),
              ("The matrix at fixed probability cuts", pd.read_csv(rules / "matrix_at_fixed_cuts.csv")),
              ("Extant results from each training run", pd.read_csv(runs / "extant_by_run.csv")),
              ("Ancestral results from each training run", pd.read_csv(runs / "ancestral_by_run.csv"))]
    title = "Supplementary Table 2 | The main results under other calling rules and from each training run"
    return blocks, title


def cluster_number():
    tables = SUPP / "cluster_number/Tables"
    blocks = [("k from 2 to 12", pd.read_csv(tables / "k_scan.csv")),
              ("Odour descriptors shuffled among the odorants", pd.read_csv(tables / "shuffled_descriptors.csv")),
              ("Re-clustering of random 80% subsamples", pd.read_csv(tables / "subsample_summary.csv")),
              ("Other clustering methods", pd.read_csv(tables / "other_methods.csv")),
              ("Odorants equally distant from two medoids", pd.read_csv(tables / "tied_odorants.csv")),
              ("Barcode distance against odour-descriptor distance, without clustering",
               pd.read_csv(tables / "mantel_barcode_descriptor.csv"))]
    title = "Supplementary Table 5 | Choice and stability of the six activation clusters"
    return blocks, title


def ancestral_confidence():
    frame = pd.read_csv(SUPP / "ancestral_confidence/Tables/ancestral_node_confidence.csv")
    title = "Supplementary Table 10 | Posterior probability and support of the 432 reconstructed ancestors"
    return [("", frame)], title


TABS = [
    ("S1 Model comparison", model_comparison),
    ("S2 Rules and runs", calling_rule_and_runs),
    ("S3 Receptors", receptors),
    ("S4 Clusters", clusters),
    ("S5 Cluster number", cluster_number),
    ("S6 Enrichment stats", enrichment),
    ("S7 Ancestral nodes", nodes),
    ("S8 Duplications", duplications),
    ("S9 Datasets", datasets),
    ("S10 Ancestral confidence", ancestral_confidence),
    ("S11 Calibration", calibration),
    ("S12 Odorants", odorants),
    ("S13 Panels", panels),
    ("S14 Chemspace controls", chemspace_controls),
]


def read_legends(path):
    """Return {table number: legend} from a plain-text legends file.

    Parameters
    ----------
    path : Path
        One paragraph per table, separated by blank lines, each starting with
        "Supplementary Table N |". Other paragraphs are ignored.

    Returns
    -------
    dict
        Table number to the whole paragraph, with its line breaks joined into spaces.
    """
    import re
    legends = {}
    for paragraph in path.read_text().split("\n\n"):
        text = " ".join(paragraph.split())
        match = re.match(r"Supplementary Table (\d+) \|", text)
        if match:
            legends[int(match.group(1))] = text
    return legends


def write_tab(book, sheet, blocks, legend):
    """Write one tab: a wrapped title or legend, then each block as its own headed sub-table.

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
    ap.add_argument("--legends", type=Path, help="legends file; without it each tab carries its title")
    args = ap.parse_args()
    legends = read_legends(args.legends) if args.legends else {}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # The published panels must be the ones the analysis actually used.
    import sys
    sys.path.insert(0, str(REPO / "Chemicals"))
    try:
        import chemspace as cs
        assert all(FG_SMARTS[k[3:]] == v for k, v in cs.FUNCTIONAL_GROUP_SMARTS.items()), \
            "SMARTS panel has drifted from the code"
        assert [d[0] for d in DESCRIPTORS_22] == cs.DESCRIPTOR_NAMES, "descriptor panel has drifted"
    except ImportError:
        print("note: chemspace not importable (needs RDKit); panels not cross-checked")

    book = Workbook()
    book.remove(book.active)                        # drop the default empty sheet
    for number, (sheet, builder) in enumerate(TABS, start=1):
        blocks, title = builder()
        write_tab(book, sheet, blocks, legends.get(number, title))
        rows = sum(len(f) for _, f in blocks)
        print(f"{sheet:24s} {len(blocks):>2} block(s), {rows:>5} rows")
    book.active = 0
    book.save(args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
