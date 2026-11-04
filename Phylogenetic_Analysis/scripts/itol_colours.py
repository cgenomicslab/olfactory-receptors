"""iTOL branch colours for the six-species trees (Figure 1c, Extended Data Fig. 1c).

All five trees have the same 584 leaves, so one file colours any of them. Colours follow
the taxid in each leaf name (<taxid>.<UniProt accession>); human leaves are split into
class I and class II with data/HumanTree/human433_OR_classes.csv.

Format: https://itol.embl.de/help/colors_styles_template.txt
In iTOL, drag figures/itol/six_species.itol_colours.txt onto the tree.

Run from Phylogenetic_Analysis/:  python scripts/itol_colours.py
"""
from pathlib import Path

import pandas as pd

TAXID_COLOUR = {
    "7955":   "#2A9D8F",   # Danio rerio
    "7740":   "#C4A01D",   # Branchiostoma lanceolatum
    "7764":   "#ab7383",   # Eptatretus burgeri
    "75743":  "#C56E4F",   # Scyliorhinus torazame
    "137246": "#C56E4F",   # Chiloscyllium punctatum
}
HUMAN_CLASS_COLOUR = {"Class_I": "#7f1734", "Class_II": "#557c99"}

classes = pd.read_csv("data/HumanTree/human433_OR_classes.csv")
human = dict(zip(classes.uniprot.str.strip(), classes.OR_class.str.strip()))

leaves = [line[1:].strip()
          for line in open("data/ReferenceTree/PF13853.9606_7955_7740_7764_75743_137246.fa")
          if line.startswith(">")]

lines = []
for leaf in leaves:
    taxid, uniprot = leaf.split(".", 1)
    colour = HUMAN_CLASS_COLOUR[human[uniprot]] if taxid == "9606" else TAXID_COLOUR[taxid]
    lines.append(f"{leaf},branch,{colour},normal,1")

out = Path("figures/itol/six_species.itol_colours.txt")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("TREE_COLORS\nSEPARATOR COMMA\nDATA\n#NODE_ID,TYPE,COLOR,STYLE,SIZE_FACTOR\n"
               + "\n".join(lines) + "\n")
print(f"{out}: {len(lines)} leaves")
