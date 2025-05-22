import numpy as np
import torch
from esm.data import Alphabet

# test_seq = "ARNDCEQGHILKMFPSTWYV"
#
# # Boltz-style aatype encoding
# boltz_order = "ARNDCQEGHILKMFPSTWYV"
# aa_to_boltz = {aa: i for i, aa in enumerate(boltz_order)}
# aatype = torch.tensor([aa_to_boltz[aa] for aa in test_seq]).unsqueeze(0)
#
# # ESM token encoding
# alphabet = Alphabet.from_architecture("ESM-1b")
# batch_converter = alphabet.get_batch_converter()
# _, _, esm_tokens = batch_converter([("test_seq", test_seq)])
#
# # Display
# print("AA sequence:      ", list(test_seq))
# print("Boltz aatype:     ", aatype[0].tolist())
# print("ESM token indices:", esm_tokens[0].tolist())
# print("ESM tokens:       ", [alphabet.get_tok(i) for i in esm_tokens[0].tolist()])

# data = np.load("../../../../train_data/rcsb_processed_targets/structures/1a00.npz")
# atoms = data["atoms"]
# print(atoms.dtype.names)
#
#
# print("Atoms fields:", data["atoms"].dtype.names)
# print("Residues fields:", data["residues"].dtype.names)

import numpy as np

import numpy as np
from pathlib import Path

p = Path("../../../../train_data/overfit_targets/structures/1a0a.npz")

with np.load(p) as npz:
    atoms    = npz["atoms"]                   # ('name','element',...,'chirality')
    residues = npz["residues"]                # ('name','res_type','res_idx','atom_idx','atom_num',...,'is_standard')

# 1. mark the non‑standard residues (= ligands, glycans, covalent modifiers, ions, etc.)
lig_mask   = residues["is_standard"] == 0

# 2. expand residue‑level indices into per‑atom indices
starts  = residues["atom_idx"][lig_mask]
counts  = residues["atom_num"][lig_mask]
lig_atom_ids = np.concatenate([np.arange(s, s+n) for s, n in zip(starts, counts)])

# 3. pick out the actual atom rows
lig_atoms = atoms[lig_atom_ids]

print("ligand residues  :", residues["name"][lig_mask])
print("number of atoms :", len(lig_atoms))
print("first five atoms:", lig_atoms[:5])