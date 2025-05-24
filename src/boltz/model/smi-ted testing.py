import numpy as np
from pathlib import Path

# Chemistry
from rdkit import Chem
from rdkit.Chem import PandasTools, rdmolfiles, rdmolops, AllChem
from rdkit.Chem import Descriptors
PandasTools.RenderImagesInAllDataFrames(True)

import json

# from boltz.model.smi_ted_light import load

print(hasattr(AllChem, "EmbedMolecule"))  # should print: True

protein_path = Path("../../../../train_data/rcsb_processed_targets/structures/1a0e.npz")

dict_path = Path("../../../../train_data/ligand_smiles_dict.json")

with open(dict_path, "r") as f:
    ligand_smiles_dict = json.load(f)

# Example usage
print("Total ligands loaded:", len(ligand_smiles_dict))
print("SMILES for HEM:", ligand_smiles_dict.get("HEM", "Not found"))

with np.load(protein_path) as sequence:
    atoms    = sequence["atoms"]                   # ('name','element',...,'chirality')
    residues = sequence["residues"]                # ('name','res_type','res_idx','atom_idx','atom_num',...,'is_standard')

    print(f"residue dict keys: {residues.dtype.names}")
    print(f"atom dict keys: {atoms.dtype.names}")

print("\n\n########## TESTING PDB TO SMILES ##########\n")

ligands = []

for i, name in enumerate(residues["name"]):
    if residues["is_standard"][i] == 0:
        smile = ligand_smiles_dict[residues["name"][i]]
        ligands.append((i, smile))




print(ligands)



smiles = []



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