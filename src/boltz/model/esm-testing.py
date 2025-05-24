import numpy as np
import torch
from esm.data import Alphabet

test_seq = "ARNDCEQGHILKMFPSTWYV"

# Boltz-style aatype encoding
boltz_order = "ARNDCQEGHILKMFPSTWYV"
aa_to_boltz = {aa: i for i, aa in enumerate(boltz_order)}
aatype = torch.tensor([aa_to_boltz[aa] for aa in test_seq]).unsqueeze(0)

# ESM token encoding
alphabet = Alphabet.from_architecture("ESM-1b")
batch_converter = alphabet.get_batch_converter()
_, _, esm_tokens = batch_converter([("test_seq", test_seq)])

# Display
print("AA sequence:      ", list(test_seq))
print("Boltz aatype:     ", aatype[0].tolist())
print("ESM token indices:", esm_tokens[0].tolist())
print("ESM tokens:       ", [alphabet.get_tok(i) for i in esm_tokens[0].tolist()])

data = np.load("../../../../train_data/rcsb_processed_targets/structures/1a00.npz")
atoms = data["atoms"]
print(atoms.dtype.names)


print("Atoms fields:", data["atoms"].dtype.names)
print("Residues fields:", data["residues"].dtype.names)