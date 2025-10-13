# psmiles_canon.py
from __future__ import annotations

import logging
import pprint
import re
from typing import Dict, Tuple, List, Optional

from rdkit import Chem
from rdkit.Chem.rdmolops import GetSymmSSSR


def add_brackets(psmiles: str) -> str:
    """Convert naked '*' to '[*]' iff there are exactly two."""
    stars_no_bracket = re.findall(r"(?<!\[)\*(?!\])", psmiles)
    if len(stars_no_bracket) == 2:
        psmiles = psmiles.replace("*", "[*]")
    return psmiles


def nb_display(mol):
    """Helper to show a molecule when debugging in notebooks."""
    print(f"SMILES: {Chem.MolToCXSmiles(mol)}")
    try:
        display(mol)
    except Exception:
        pass


def get_mol(psmiles) -> Chem.RWMol:
    """Returns an RDKit RWMol from SMILES."""
    return Chem.RWMol(Chem.MolFromSmiles(psmiles))


def get_connection_info(mol=None, symbol="*") -> Dict:
    """
    Extracts: star indices/types, neighbor indices/types/bonds, shortest path
    between the neighbor atoms, stereochem and ring info.
    """
    ret_dict = {}

    stars_indices, stars_type, all_symbols, all_index = [], [], [], []
    for star_idx, atom in enumerate(mol.GetAtoms()):
        all_symbols.append(atom.GetSymbol())
        all_index.append(atom.GetIdx())
        if symbol in atom.GetSymbol():
            stars_indices.append(star_idx)
            stars_type.append(atom.GetSmarts())

    # guard: need exactly two stars
    if len(stars_indices) != 2:
        ret_dict["star"] = {"index": stars_indices, "atom_type": stars_type, "bond_type": None}
        # fill minimal structure and return
        ret_dict["symbols"] = all_symbols
        ret_dict["index"] = all_index
        ret_dict["neighbor"] = {"index": [[], []], "atom_type": [[], []], "bond_type": [[], []], "path": None}
        ring_info = mol.GetRingInfo()
        ret_dict["atom_rings"] = ring_info.AtomRings()
        ret_dict["bond_rings"] = ring_info.BondRings()
        ret_dict["stereo"] = []
        return ret_dict

    stars_bond = mol.GetBondBetweenAtoms(stars_indices[0], stars_indices[1])
    if stars_bond:
        stars_bond = stars_bond.GetBondType()

    ret_dict["symbols"] = all_symbols
    ret_dict["index"] = all_index

    ret_dict["star"] = {
        "index": stars_indices,
        "atom_type": stars_type,
        "bond_type": stars_bond,
    }

    # neighbors of each star (indices, types, and star–neighbor bond types)
    neighbor_indices = [
        [x.GetIdx() for x in mol.GetAtomWithIdx(stars_indices[0]).GetNeighbors()],
        [x.GetIdx() for x in mol.GetAtomWithIdx(stars_indices[1]).GetNeighbors()],
    ]
    neighbors_type = [
        [mol.GetAtomWithIdx(x).GetSmarts() for x in neighbor_indices[0]],
        [mol.GetAtomWithIdx(x).GetSmarts() for x in neighbor_indices[1]],
    ]
    neighbor_bonds = [
        [
            mol.GetBondBetweenAtoms(stars_indices[0], x).GetBondType()
            for x in neighbor_indices[0]
        ],
        [
            mol.GetBondBetweenAtoms(stars_indices[1], x).GetBondType()
            for x in neighbor_indices[1]
        ],
    ]

    s_path = None
    if neighbor_indices[0] and neighbor_indices[1] and neighbor_indices[0][0] != neighbor_indices[1][0]:
        s_path = Chem.GetShortestPath(
            mol, neighbor_indices[0][0], neighbor_indices[1][0]
        )

    ret_dict["neighbor"] = {
        "index": neighbor_indices,
        "atom_type": neighbors_type,
        "bond_type": neighbor_bonds,
        "path": s_path,
    }

    # Stereo info
    stereo_info = []
    for b in mol.GetBonds():
        bond_type = b.GetStereo()
        if bond_type != Chem.rdchem.BondStereo.STEREONONE:
            idx = [b.GetBeginAtomIdx(), b.GetEndAtomIdx()]
            neigh_idx = b.GetStereoAtoms()
            stereo_info.append(
                {
                    "bond_type": bond_type,
                    "atom_idx": idx,
                    "bond_idx": b.GetIdx(),
                    "neighbor_idx": list(neigh_idx),
                }
            )
    ret_dict["stereo"] = stereo_info

    # Ring info
    ring_info = mol.GetRingInfo()
    ret_dict["atom_rings"] = ring_info.AtomRings()
    ret_dict["bond_rings"] = ring_info.BondRings()

    return ret_dict


def reduce_multiplication(psmiles: str) -> str:
    r"""
    Reduces obvious repeated literal substrings between stars.
    Examples:
        [*]CCC[*]   -> [*]C[*]
        [*]COCO[*]  -> [*]CO[*]
        [*]COCCOCCOCCOC[*] -> [*]COCCOC[*]
    """
    sm = psmiles

    # skip if ring indices exist
    if re.findall(r"\d", sm):
        return psmiles

    # Polyethylene special-case
    pe_test = list(set(sm.replace("[*]", "").upper()))
    if len(pe_test) == 1 and pe_test[0] == "C":
        return "[*]C[*]"

    # all other cases
    d = {}
    for sublen in range(1, int(len(sm) / 2)):
        for i in range(0, len(sm) - sublen):
            sub = sm[i : i + sublen]
            cnt = sm.count(sub)
            if cnt >= 2 and sub not in d:
                if "[" not in sub and "]" not in sub and "*" not in sub:
                    d[sub] = cnt
    if not d:
        return psmiles

    longest_string = sorted(d, key=lambda k: len(k), reverse=True)[0]
    check_s = sm.replace("[*]", "").replace(longest_string, "")
    if not check_s:
        sm = sm.replace(longest_string, "", d[longest_string] - 1)
    return sm


def get_index_to_break(mol: Chem.RWMol, info: Dict) -> Tuple[int, int]:
    """
    Find two connected atoms to break (not part of original rings that must be preserved).
    Uses labeled N1/N2 atoms placed earlier.
    """
    # Find labeled atoms
    n_idx: List[int] = []
    for atom in mol.GetAtoms():
        if "atomLabel" in atom.GetPropsAsDict() and "N" in atom.GetProp("atomLabel"):
            n_idx.append(atom.GetIdx())
    if len(n_idx) != 2:
        raise UserWarning("Canonicalization failed (no N1/N2 labels).")

    # Remove N1–N2 temporarily to compute original rings
    bnd = mol.GetBondBetweenAtoms(n_idx[0], n_idx[1]).GetBondType()
    mol.RemoveBond(n_idx[0], n_idx[1])
    GetSymmSSSR(mol)
    original_atom_rings = mol.GetRingInfo().AtomRings()
    original_bond_rings = mol.GetRingInfo().BondRings()

    # Add bond back
    mol.AddBond(n_idx[0], n_idx[1], bnd)
    GetSymmSSSR(mol)
    new_atom_rings = mol.GetRingInfo().AtomRings()
    new_bond_rings = mol.GetRingInfo().BondRings()

    # desired ring length = neighbor path length (if available)
    ring_length_to_break = len(info["neighbor"]["path"]) if info.get("neighbor", {}).get("path") else None

    rings_match, bond_rings_match = [], []
    for atom_ring, bond_ring in zip(new_atom_rings, new_bond_rings):
        cond_core = set(atom_ring).issuperset(set(n_idx))
        cond_len  = (ring_length_to_break is None) or (len(atom_ring) == ring_length_to_break)
        if cond_core and cond_len:
            rings_match.append(atom_ring)
            bond_rings_match.append(bond_ring)

    if len(rings_match) == 0:
        # relax: accept any ring containing N1/N2
        for atom_ring, bond_ring in zip(new_atom_rings, new_bond_rings):
            if set(atom_ring).issuperset(set(n_idx)):
                rings_match.append(atom_ring)
                bond_rings_match.append(bond_ring)

    def bond_in_original_rings(bond_index):
        for _ring in original_bond_rings:
            if bond_index in _ring:
                return True
        return False

    def get_break_idx(ring_to_break):
        sorted_ring = sorted(ring_to_break)
        for idx_1 in sorted_ring:
            possible_neighbors = sorted([x.GetIdx() for x in mol.GetAtomWithIdx(int(idx_1)).GetNeighbors()])
            for idx_2 in possible_neighbors:
                bond = mol.GetBondBetweenAtoms(idx_1, idx_2)
                if bond is None:
                    continue
                bond_idx = bond.GetIdx()
                if idx_2 in sorted_ring and not bond_in_original_rings(bond_idx):
                    return idx_1, idx_2
        raise UserWarning("Canonicalization failed.")

    rings_match = sorted(rings_match, key=sum)
    for ring_to_break in rings_match:
        break_idx_1, break_idx_2 = get_break_idx(ring_to_break)
        if break_idx_1 is not None and break_idx_2 is not None:
            break

    # if triangle ring and large molecule: break at labeled atoms
    if len(ring_to_break) == 3 and len(info["symbols"]) > 5:
        break_idx_1 = n_idx[0]
        break_idx_2 = n_idx[1]

    return break_idx_1, break_idx_2


def unify(psmiles: str) -> str:
    r"""
    “Unify” a PSMILES by:
      - connecting the two star-neighbors with the star-neighbor bond type,
      - removing stars,
      - canonicalizing the now cyclic molecule,
      - breaking a ring bond and re-adding stars at that position.
    """

    # Show atom indices when DEBUG
    if logging.DEBUG >= logging.root.level:
        from rdkit.Chem.Draw import IPythonConsole
        IPythonConsole.drawOptions.addAtomIndices = True

    mol = get_mol(psmiles)
    info = get_connection_info(mol)
    logging.debug(f"(1) Labels + connection dict\n{pprint.pformat(info)}")

    # SPECIAL CASES
    if psmiles == "[*]C[*]":
        logging.debug("Found [*]C[*]; returning [*]C[*]")
        return Chem.MolToSmiles(mol)

    if info["neighbor"]["path"] and len(info["neighbor"]["path"]) == 2:
        logging.debug("Neighbors already connected; return unified here.")
        return Chem.MolToSmiles(mol)

    if info["neighbor"]["index"][0] and info["neighbor"]["index"][1] and \
       (info["neighbor"]["index"][0][0] == info["neighbor"]["index"][1][0]):
        logging.debug("Neighbors are the same atom; return unified here.")
        return Chem.MolToSmiles(mol)

    # label neighbors as N1/N2 for ring breaking later
    if info["neighbor"]["index"][0] and info["neighbor"]["index"][1]:
        mol.GetAtomWithIdx(info["neighbor"]["index"][0][0]).SetProp("atomLabel", "N1")
        mol.GetAtomWithIdx(info["neighbor"]["index"][1][0]).SetProp("atomLabel", "N2")
    if logging.DEBUG >= logging.root.level:
        nb_display(mol)

    # (2) connect neighbors
    logging.debug("(2) Connect neighbors with star-neighbor bond type(s)")
    if not info["neighbor"]["index"][0] or not info["neighbor"]["index"][1]:
        # not enough info to connect so we bail out gracefully
        return Chem.MolToSmiles(mol)

    bt0 = info["neighbor"]["bond_type"][0][0] if info["neighbor"]["bond_type"][0] else Chem.BondType.SINGLE
    bt1 = info["neighbor"]["bond_type"][1][0] if info["neighbor"]["bond_type"][1] else Chem.BondType.SINGLE
    # prefer bt0; fall back to SINGLE if types disagree
    btype = bt0 if bt0 == bt1 else Chem.BondType.SINGLE

    mol.AddBond(
        info["neighbor"]["index"][0][0],
        info["neighbor"]["index"][1][0],
        btype,
    )
    if logging.DEBUG >= logging.root.level:
        nb_display(mol)

    # (3) remove stars + those bonds
    logging.debug("(3) Remove stars and their bonds")
    if info["star"]["index"]:
        try:
            if info["neighbor"]["index"][0]:
                mol.RemoveBond(info["star"]["index"][0], info["neighbor"]["index"][0][0])
            if info["neighbor"]["index"][1]:
                mol.RemoveBond(info["star"]["index"][1], info["neighbor"]["index"][1][0])
        except Exception:
            pass
        # remove higher index first
        for sid in sorted(info["star"]["index"], reverse=True):
            try:
                mol.RemoveAtom(sid)
            except Exception:
                pass
    if logging.DEBUG >= logging.root.level:
        nb_display(mol)

    # (4) canonicalize ring
    logging.debug("(4) Canonicalize cyclic intermediate")
    Chem.Kekulize(mol, clearAromaticFlags=True)
    sm = Chem.MolToCXSmiles(mol)
    mol = Chem.RWMol(Chem.MolFromSmiles(sm))
    Chem.Kekulize(mol, clearAromaticFlags=True)
    Chem.Kekulize(mol, clearAromaticFlags=True) # idempotence guard
    sm = Chem.MolToCXSmiles(mol)
    mol = Chem.RWMol(Chem.MolFromSmiles(sm))
    if logging.DEBUG >= logging.root.level:
        nb_display(mol)

    # (5) choose a ring bond to break
    logging.debug("(5) Find bond to break")
    Chem.Kekulize(mol, clearAromaticFlags=True)
    b0_break, b1_break = get_index_to_break(mol, info)
    btype_removed = mol.GetBondBetweenAtoms(b0_break, b1_break).GetBondType()
    if btype_removed == Chem.rdchem.BondType.AROMATIC:
        conj = mol.GetBondBetweenAtoms(b0_break, b1_break).GetIsConjugated()
        btype_removed = Chem.rdchem.BondType.DOUBLE if conj else Chem.rdchem.BondType.SINGLE

    logging.debug(f"Break bond between {b0_break} and {b1_break} (type {btype_removed})")

    mol.RemoveBond(b0_break, b1_break)
    if logging.DEBUG >= logging.root.level:
        nb_display(mol)

    # (6) reintroduce stars at break, renumber
    logging.debug("(6) Add stars at the break + renumber")
    idx = mol.AddAtom(Chem.AtomFromSmarts("*")); mol.AddBond(b0_break, idx, btype_removed)
    idx = mol.AddAtom(Chem.AtomFromSmarts("*")); mol.AddBond(b1_break, idx, btype_removed)

    sm = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(sm)
    if logging.DEBUG >= logging.root.level:
        nb_display(mol)

    return Chem.MolToSmiles(mol)


def canonicalize(psmiles: str) -> str:
    """
    Full pipeline:
      (1) unify once (connect --> canonize ring --> reopen)
      (2) add brackets if needed
      (3) iteratively reduce literal repeats (<=20 passes)
      (4) add brackets again
    """
    new_ps = unify(psmiles)
    new_ps = add_brackets(new_ps)

    for _ in range(20):
        new = reduce_multiplication(new_ps)
        new = add_brackets(new)
        if new == new_ps:
            break
        new_ps = new

    return add_brackets(new_ps)
