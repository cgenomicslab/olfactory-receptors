"""The functional-group panel used by every analysis in the repository.

Each group is a SMARTS pattern; a molecule carries the group if RDKit finds at least one
match. Fig. 2, Fig. 3 and Fig. 4 all take their groups from this dictionary, so a group
means the same thing in every figure and table.
"""
from rdkit import Chem

FUNCTIONAL_GROUPS = {
    # the acid or its carboxylate: some odorants are recorded as the anion
    "carboxylic_acid": "[CX3](=O)[OX2H1,OX1-]",
    # acyl carbon bonded to C or H, so carbonates and carbamates are excluded
    "ester": "[CX3;$([CX3][#6]),$([CX3H1])](=O)[OX2H0][#6]",
    # an ester whose C(=O)-O bond is in a ring
    "lactone": "[#6]@[OX2H0]@[CX3](=O)",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "alcohol": "[CX4][OX2H]",
    "phenol": "c[OX2H]",
    # the oxygen of an ester or lactone is not an ether oxygen
    "ether": "[OD2;!$(O[CX3]=[O,S])]([#6])[#6]",
    # neutral trivalent N, not amide-like and not bonded to N, O or S
    "amine": "[NX3;+0;!$(N[#6]=[O,S,N]);!$(N[!#6;!#1])]",
    "amide": "[NX3][CX3](=[OX1])[#6]",
    "thiol": "[#16X2H]",
    # non-aromatic divalent sulfur bonded to carbon, without H (thioethers, disulfides)
    "sulfide": "[SX2H0][#6]",
    "sulfur": "[#16]",
    "nitrile": "[NX1]#[CX2]",
    "halogen": "[F,Cl,Br,I]",
    # any aromatic ring: benzenes, pyridines, pyrazines, furans, thiophenes, pyrroles
    "aromatic_ring": "[a]",
    "alkene": "[CX3]=[CX3]",
    # isopropyl or isopropylidene, the branch end of an isoprene unit
    "terpene_isopropyl": "[$([CH3][CX4H1]([CH3])[#6]),$([CH3][CX3]([CH3])=[CX3])]",
}

PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in FUNCTIONAL_GROUPS.items()}


def label(name):
    """Return the display label of a group, e.g. 'carboxylic acid'."""
    return name.replace("_", " ")


def has_group(mol, name):
    """Return True if the RDKit molecule carries the functional group."""
    return mol is not None and mol.HasSubstructMatch(PATTERNS[name])
