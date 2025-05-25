from typing import Union, Optional
import torch
import torch.nn as nn

from boltz.model.esm.data import Alphabet, BatchConverter
from boltz.model.esm.modules import ContactPredictionHead, ESM1bLayerNorm, RobertaLMHead, TransformerLayer

import json
import numpy as np

from pathlib import Path

class Translate():
    def __init__(self):
        self.alphabet = Alphabet.from_architecture("ESM-1b")
        self.batch_converter = self.alphabet.get_batch_converter()

        CUR_DIR = Path(__file__).resolve().parent
        RES_JSON = CUR_DIR / "res_type_to_aa.json"
        LIG_JSON = CUR_DIR / "ligand_to_smiles.json"

        with RES_JSON.open() as f:
            res_type_to_aa = json.load(f)
        self.res_type_to_aa = {int(k): v for k, v in res_type_to_aa.items()}

        with LIG_JSON.open() as f:
            self.ligand_to_smiles = json.load(f)

    def boltz_to_esm(
            self, residues: np.ndarray, max_len: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        aa_list: list[str] = []
        indices: list[int] = []
        ligand_skips = 0

        for i in range(len(residues)):
            if max_len is not None and i >= max_len:
                break
            if residues["is_standard"][i]:
                res_id = int(residues["res_type"][i])
                aa_list.append(self.res_type_to_aa.get(res_id, "X"))
                indices.append(i + ligand_skips)
            else:
                ligand_skips += int(residues["atom_num"][i])

        if not aa_list:
            raise ValueError("No standard amino‑acids found in this structure.")

        _, _, tokens = self.batch_converter([("protein", "".join(aa_list))])
        tokens = tokens.squeeze(0)

        esm_indices = torch.tensor(indices, dtype=torch.long)

        return tokens, esm_indices

    def esm_to_boltz(
            self,
            esm_output: dict,
            batched_idx: torch.Tensor,  # (B, N_max)
            full_seq_len: int,
    ) -> torch.Tensor:
        last = max(esm_output["representations"])
        esm_emb = esm_output["representations"][last]  # (B, T, D)
        B, _, D = esm_emb.shape
        device = esm_emb.device

        out = torch.zeros((B, full_seq_len, D), device=device)
        for b in range(B):
            idx = batched_idx[b]  # shape (N_max,)
            if idx.numel() == 0:
                continue
            res_emb = esm_emb[b, 1:-1][: idx.numel()]  # (n_res, D)
            out[b, idx] = res_emb

        return out

    def boltz_to_smi_ted(
            self,
            residues: np.ndarray,
            max_len: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        smiles: list[str] = []
        span_tuples: list[tuple[int, int]] = []

        # print(f"[Translate] residues.shape: {residues.shape}")
        # print(f"[Translate] total residues: {len(residues)}")
        #
        # non_standard_mask = ~residues["is_standard"]
        # non_standard_indices = np.nonzero(non_standard_mask)[0].tolist()
        #
        # print(f"[Translate] non-standard count: {non_standard_mask.sum()}")
        # print(f"[Translate] non-standard indices: {non_standard_indices}")

        ligand_atoms_already = 0
        for i, name in enumerate(residues["name"]):
            if max_len is not None and i >= max_len:
                break
            if not residues["is_standard"][i]:
                smile = self.ligand_to_smiles.get(name)
                # print(f"  Residue {i} — name: {name} | SMILES found: {smile is not None}")
                if smile:
                    num_atoms = int(residues["atom_num"][i])
                    start = i + ligand_atoms_already
                    end = start + num_atoms
                    span_tuples.append((start, end))
                    smiles.append(smile)
                    ligand_atoms_already += num_atoms
                    # print(f"    → span: ({start}, {end})")

        if not smiles:
            # print("[Translate] No SMILES strings found.")
            return (torch.zeros((0, 0), dtype=torch.long),
                    torch.zeros((0, 2), dtype=torch.long))

        max_len = max(len(s) for s in smiles)
        smiles_ascii = torch.zeros(len(smiles), max_len, dtype=torch.long)
        for row, smile in enumerate(smiles):
            smiles_ascii[row, : len(smile)] = torch.tensor([ord(c) for c in smile])

        smi_ted_indices = torch.tensor(span_tuples, dtype=torch.long)

        # print(f"[Translate] Final: {len(smiles)} SMILES | span_tuples: {span_tuples}")

        return smiles_ascii, smi_ted_indices

    def smi_ted_to_boltz(
        self,
        smi_ted_emb: torch.Tensor,   # (num_lig, dimensions)
        ligand_spans: torch.Tensor,  # (num_lig, 2), where the second dimension holds the start/end token‑indices
        full_seq_len: int,
    ) -> torch.Tensor:
        if smi_ted_emb.numel() == 0:
            return torch.zeros(full_seq_len, 0, device=smi_ted_emb.device)

        dimensions = smi_ted_emb.size(-1)
        out = torch.zeros(full_seq_len, dimensions, device=smi_ted_emb.device)

        # broadcast to all atom tokens
        for row, (start, end) in enumerate(ligand_spans.tolist()):
            out[start:end] = smi_ted_emb[row]

        return out

    def ascii_tensor_to_smiles(self, smiles_ascii: torch.Tensor) -> list[str]:
        """
        Convert a zero‑padded tensor of ASCII code‑points back to a list of SMILES strings.

        Parameters
        ----------
        smiles_ascii : torch.Tensor
            `int64` / `int32` tensor of shape **(N, L)** where
            • `N` = number of ligands
            • `L` = max string length in the batch
            Each row contains the Unicode/ASCII code‑points of the SMILES characters,
            padded with zeros on the right.

        Returns
        -------
        list[str]
            `N` Python strings, one per ligand.
        """
        if smiles_ascii.ndim != 2:
            raise ValueError("Expected a 2‑D tensor of shape (N, L)")

        smiles_list: list[str] = []
        for row in smiles_ascii.cpu():  # iterate over ligands
            # Convert to Python list, truncate at first padding‑zero
            codes = row.tolist()
            if 0 in codes:
                codes = codes[: codes.index(0)]
            # Convert each code‑point back to its character and join
            smiles_list.append("".join(chr(c) for c in codes))

        return smiles_list