from ete4 import PhyloTree
from ete4.smartview import Layout, BASIC_LAYOUT

t = PhyloTree(open("PF13853_9606.7764.mafft.lg.treefile.rooted.withinternalnames"), parser=1)

outgroup_leaf_name = 'New|7764.A0A8C4N3S4'
t.set_outgroup(outgroup_leaf_name)

taxid_color_map = {
    "9606": "#b97171",   # human – rose
    "7764": "#0057da",   # hagfish – dark blue (out‑group)

}

STYLE_CL1 = {'stroke-width': 3, 'stroke': 'LightSteelBlue'}
STYLE_CL2 = {'stroke-width': 3, 'stroke': 'Moccasin'}

branch_styles = {}

for n in t.traverse():
    if n.is_leaf:
        header = n.name
        if header.startswith('New|'):
            header = header.split('|', 1)[1]   
        taxid = header.split('.', 1)[0]
        if taxid in taxid_color_map:
            colour = taxid_color_map[taxid]
            branch_styles[n] = {'stroke-width': 2, 'stroke': colour}

node1 = None
for n in t.traverse():
    if n.name == 'node_1':
        node1 = n
        break
assert node1 and len(node1.children) == 2, "node_1 must have exactly two direct children"

clade_styles = {node1.children[0]: STYLE_CL1,
                node1.children[1]: STYLE_CL2}

for clade, style in clade_styles.items():
    for parent in clade.traverse():
        for child in parent.children:
            branch_styles[child] = style   


def draw_node(node):
    s = branch_styles.get(node)
    return {'hz-line': s, 'vt-line': s} if s else {}

layout = Layout('TaxID + node₁ subtrees', draw_node=draw_node)

t.explore(layouts=[layout])