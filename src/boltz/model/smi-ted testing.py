# import numpy as np
# from pathlib import Path
#
# # Chemistry
# from rdkit import Chem
# from rdkit.Chem import PandasTools, rdmolfiles, rdmolops, AllChem
# from rdkit.Chem import Descriptors
# PandasTools.RenderImagesInAllDataFrames(True)
#
# import json
#
# # from boltz.model.smi_ted_light import load
#
# print(hasattr(AllChem, "EmbedMolecule"))  # should print: True
#
# protein_path = Path("../../../../train_data/rcsb_processed_targets/structures/1a0e.npz")
#
# dict_path = Path("../../../../train_data/ligand_smiles_dict.json")
#
# with open(dict_path, "r") as f:
#     ligand_smiles_dict = json.load(f)
#
# # Example usage
# print("Total ligands loaded:", len(ligand_smiles_dict))
# print("SMILES for HEM:", ligand_smiles_dict.get("HEM", "Not found"))
#
# with np.load(protein_path) as sequence:
#     atoms    = sequence["atoms"]                   # ('name','element',...,'chirality')
#     residues = sequence["residues"]                # ('name','res_type','res_idx','atom_idx','atom_num',...,'is_standard')
#
#     print(f"residue dict keys: {residues.dtype.names}")
#     print(f"atom dict keys: {atoms.dtype.names}")
#
# print("\n\n########## TESTING PDB TO SMILES ##########\n")
#
# smiles = []
#
# for i, name in enumerate(residues["name"]):
#     if residues["is_standard"][i] == 0:
#         smile = ligand_smiles_dict[residues["name"][i]]
#         smiles.append((i, smile))
#
# print(smiles)
#
# print("\n\n########## TESTING PDB TO ESM ##########\n")
#
# three_to_one = {
#     "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
#     "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
#     "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
#     "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
#     "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
# }
#
# res_map = {}
#
# for residue in residues:
#     if residue["is_standard"]:
#         res_type = residue["res_type"]
#         name = residue["name"].decode() if isinstance(residue["name"], bytes) else residue["name"]
#         if res_type not in res_map:
#             res_map[res_type] = three_to_one[name]
#
# # Print the mapping sorted by res_type for readability
# for res_type in sorted(res_map):
#     print(f"{res_type:2d} → {res_map[res_type]}")
#
# res_type_to_aa = {
#     2: "A", 3: "R", 4: "N", 5: "D", 6: "C", 7: "Q", 8: "E", 9: "G",
#     10: "H", 11: "I", 12: "L", 13: "K", 14: "M", 15: "F", 16: "P",
#     17: "S", 18: "T", 19: "W", 20: "Y", 21: "V"
# }

# import json
#
# with open("residue_to_aa.json", "w") as f:
#     json.dump(res_type_to_aa, f, indent=2)





# ligand_smiles_dict = load_ligand_dict_from_pickle("../../../../train_data/ligands.pkl")
#
# print(list(ligand_smiles_dict.keys())[:10])

# direct_smiles = []
#
# for ligand in ligands:
#     ligand_name = ligand["name"].tobytes().decode("utf-8").strip()
#     if ligand_name in ligand_smiles_dict:
#         direct_smiles.append(ligand_smiles_dict[ligand_name])
#     else:
#         print(f"Warning: Ligand {ligand_name} not found in SMILES dictionary.")
#         direct_smiles.append(None)
#
# print(f"smiles strings: {direct_smiles}")



#
# def build_and_save_ligand_dict(sdf_path, output_path):
#     from rdkit import Chem
#     ligand_dict = {}
#     suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
#     for mol in suppl:
#         if mol is None:
#             continue
#         try:
#             pdb_id = mol.GetProp("_Name").strip().upper()
#             smi = Chem.MolToSmiles(mol)
#             ligand_dict[pdb_id] = smi
#         except Exception:
#             continue
#     with open(output_path, "wb") as f:
#         pickle.dump(ligand_dict, f)
#     print(f"Saved {len(ligand_dict)} ligands.")
#
#
# build_and_save_ligand_dict("../../../../train_data/components-pub.sdf", "../../../../train_data/ligands.pkl")


"""
pytest‑compatible unit test for the ascii_tensor_to_smiles helper.

Run with:
    python -m pytest test_smiles_ascii.py
or simply execute the file with Python to see a quick PASS/FAIL message.
"""

import numpy as np
import torch
from Bio.Seq import translate

from ..data.feature.translate import Translate           # your class


def make_mock_residues(smiles_list):
    """
    Build a minimal structured array that mimics the fields
    Translate.boltz_to_smi_ted() expects: name, is_standard,
    atom_idx, atom_num.
    """
    dtype = [
        ("name", "U8"),
        ("is_standard", "i1"),
        ("atom_idx", "i4"),
        ("atom_num", "i4"),
    ]

    recs = []
    cursor = 0
    for s in smiles_list:
        # Non‑standard residue (ligand)  → is_standard = 0
        recs.append((s, 0, cursor, len(s)))
        cursor += len(s)

    return np.array(recs, dtype=dtype)


def test_ascii_roundtrip():
    # ----- 1. example ligands ---------------------------------------------
    ligands = ["ATP", "FAD", "NAD"]
    print("Ligand names:", ligands)

    # ----- 2. lookup SMILES ------------------------------------------------
    translator = Translate()
    expected_smiles = [translator.ligand_to_smiles[name] for name in ligands]
    print("SMILES strings (from lookup):", expected_smiles)

    # ----- 3. encode to ASCII tensor ---------------------------------------
    residues = make_mock_residues(ligands)
    smiles_ascii, spans = translator.boltz_to_smi_ted(residues)
    print("Encoded ASCII tensor:\n", smiles_ascii)

    # ----- 4. decode back to SMILES ---------------------------------------
    decoded = Translate.ascii_tensor_to_smiles(smiles_ascii)
    print("Decoded SMILES strings:", decoded)

    # ----- 5. compare ------------------------------------------------------
    assert decoded == expected_smiles, f"Round-trip failed:\n{decoded}\n!=\n{expected_smiles}"
    print("✓ SMILES ASCII round‑trip passed!")


if __name__ == "__main__":
    test_ascii_roundtrip()