# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union
import torch
import torch.nn as nn

from boltz.model.esm.data import Alphabet, BatchConverter
from boltz.model.esm.modules import ContactPredictionHead, ESM1bLayerNorm, RobertaLMHead, TransformerLayer


class ESM2(nn.Module):
    def __init__(
        self,
        num_layers: int = 10,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        alphabet: Union[Alphabet, str] = "ESM-1b",
        token_dropout: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        if not isinstance(alphabet, Alphabet):
            alphabet = Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.token_dropout = token_dropout

        self._init_submodules()

    def _init_submodules(self):
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    add_bias_kv=False,
                    use_esm1b_layer_norm=True,
                    use_rotary_embeddings=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.contact_head = ContactPredictionHead(
            self.num_layers * self.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)

        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

    def forward(self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)

        if self.token_dropout:
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]


class Translate():
    def __init__(self):
        self.alphabet = Alphabet.from_architecture("ESM-1b")
        self.batch_converter = self.alphabet.get_batch_converter()

        self.boltz_order = "ARNDCQEGHILKMFPSTWYV"
        self.aa_to_boltz = {aa: i for i, aa in enumerate(self.boltz_order)}

    def aa_to_esm(self, sequence_batch):
        # convert whole batch at once to esm
        return self.batch_converter(sequence_batch)

    def esm_to_boltz(self, tokens):
        boltz_tensor = torch.full_like(tokens, fill_value=-1)  # -1 for unknowns

        # Loop through tokens and convert only standard amino acids
        for tok_id in torch.unique(tokens):
            aa = self.alphabet.get_tok(tok_id.item())
            if aa in self.aa_to_boltz:
                boltz_index = self.aa_to_boltz[aa]
                boltz_tensor[tokens == tok_id] = boltz_index

        return boltz_tensor

    def boltz_to_esm(self, aatype_tensor):
        boltz_to_aa = {v: k for k, v in self.aa_to_boltz.items()}
        batch_size, seq_len = aatype_tensor.shape
        seqs = []
        for i in range(batch_size):
            seq = "".join([boltz_to_aa.get(idx.item(), "X") for idx in aatype_tensor[i]])
            seqs.append(("reconstructed", seq))
        return self.aa_to_esm(seqs)

    def esm_print(self, seqs):
        # Get ESM tokens
        labels, strings, esm_tokens = self.aa_to_esm(seqs)
        boltz_tensor = self.esm_to_boltz(esm_tokens)

        # Decode token IDs to readable strings (e.g., <cls>, A, R, N, ..., <eos>)
        decoded_tokens = [self.alphabet.get_tok(tok.item()) for tok in esm_tokens[0]]

        # Print results
        print("Input sequence:  ", strings[0])
        print("ESM token IDs:   ", esm_tokens[0].tolist())
        print("ESM token string:", decoded_tokens)
        print("Boltz-style:     ", boltz_tensor[0].tolist())