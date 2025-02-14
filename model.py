import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
import os

from msa_modules import AxialTransformerLayer


class PhyloATTN(nn.Module):
    def __init__(self, cfgs):
        super().__init__()

        self.vocab_size = cfgs.model.vocab_size
        self.patch_size = cfgs.model.patch_size
        self.patch_num =  cfgs.model.fixed_length // self.patch_size
        self.embed_dim = cfgs.model.embed_dim
        self.encoder_attn_layers = cfgs.model.encoder_attn_layers
        self.num_enc_heads = cfgs.model.num_enc_heads

        self.num_enc_layers = cfgs.model.num_enc_layers
        self.dropout = 0.4
        # Encoder layer
        self.seq_emb_layers = nn.ModuleList(
            [
                AxialTransformerLayer(
                    self.embed_dim,
                    self.embed_dim * 4,
                    self.num_enc_heads,
                    self.dropout,
                    self.dropout,
                    self.dropout,
                    1024
                )
                for _ in range(self.num_enc_layers)
            ]
        )
        self.embed = nn.Sequential(
            nn.Linear(self.vocab_size * self.patch_size, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )


        # __________________aggregate layer__________________
        self.h_linear_last = nn.Linear(self.embed_dim, self.embed_dim)

        self.g_linear_last = nn.Linear(self.embed_dim, self.embed_dim)
        self.g_attn_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.g_attn_k = nn.Linear(self.embed_dim, self.embed_dim)
        # __________________aggregate layer end__________________


        # decoder layer
        self.s_out = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, 1),
        )


    def model_params(self):
        return list(self.parameters())


    def encode_zxr(self, batch_input, batch_seq_mask=None):
        batch_input = batch_input.float()

        batch_size, num_rows, num_cols, embed_dim = batch_input.size()

        self.patch_num = math.ceil(num_cols / self.patch_size)

        c = self.patch_num

        x = einops.rearrange(batch_input, 'b r (c k) e -> b r c (k e)', k=self.patch_size)
        x = self.embed(x)

        batch_seq_mask_ = einops.repeat(batch_seq_mask[:,::self.patch_size], 'b s -> b n s', n=num_rows)

        x = einops.rearrange(x, 'b r c d -> r c b d')
        for layer in self.seq_emb_layers:
            x = layer(x, self_attn_padding_mask=batch_seq_mask_)
        
        # use col_attentions
        x = einops.rearrange(x, 'r c b d -> b r c d')

        return x

    def decode_gg(self, x_i, x_j, seq_mask, ij_indices):

        self.seq_mask = seq_mask

        x = self.aggregate(x_i, x_j, ij_indices)
        scores = self.s_out(x).squeeze(-1)
        scores = scores * seq_mask
        scores = scores.sum(-1)

        return scores


    def aggregate(self, x_i, x_j, ij_indices, batchwise_ij_indices=False):
        seq_mask = self.seq_mask

        h = self.h_linear_last(x_i - x_j)

        z = torch.sigmoid(h)
        x = (z * x_i + (1 - z) * x_j) # B R C D


        if self.batch_input.size(1) > 2:
            q = self.g_attn_q(x)
            k = self.g_attn_k(self.batch_input)
            v = self.batch_input

            # r: subtrees num, n: ij pairs num
            # b n r
            alpha = torch.einsum('bncd,brcd->bnr', q, k) / math.sqrt(self.embed_dim * self.patch_num)

            i_indices, j_indices = ij_indices

            if batchwise_ij_indices:
                b_indices = torch.arange(alpha.shape[0]).to(alpha.device)
                alpha[b_indices, :, i_indices] += float('-inf')
                alpha[b_indices, :, j_indices] += float('-inf')
            else:
                # fst step
                if i_indices.dim() == 1:
                    n_indices = torch.arange(alpha.size(1)).to(alpha.device)
                    alpha[:, n_indices, i_indices] += float('-inf')
                    alpha[:, n_indices, j_indices] += float('-inf')
                # snd step -> end
                elif i_indices.dim() == 2:
                    b, n, r = alpha.shape

                    b_indices = torch.arange(alpha.shape[0]).to(alpha.device)
                    b_indices = b_indices[:, None].expand(b, n)
                    n_indices = torch.arange(alpha.size(1)).to(alpha.device)
                    n_indices = n_indices[None, :].expand(b, n)
                    
                    alpha[b_indices, n_indices, i_indices] += float('-inf')
                    alpha[b_indices, n_indices, j_indices] += float('-inf')
                else:
                    assert False

            alpha = torch.softmax(alpha, dim=-1)

            x_global_res = torch.einsum('bnr,brcd->bncd', alpha, v)

            g = self.g_linear_last(x_global_res)

            w = torch.sigmoid(g)
            x = (1 - w) * x + w * x_global_res

        return x


    def decode_zxr(self, batch_input, batch_seq_mask=None, indices_to_prev_info=None):

        batch_size, num_rows, num_cols = batch_input.size()[:3]

        actions_ij_prev, score_indices_to_prev, logits_prev = indices_to_prev_info

        seq_mask = ~ batch_seq_mask[:, None, ::self.patch_size]

        self.batch_input = batch_input

        if logits_prev is None:
            x = batch_input
            # Cartesian product on row
            # (b r 1 c d), (b 1 r c d) -> (b r r c 2d)

            x_i = einops.repeat(x, 'b r c d -> b r x c d', x=num_rows)
            x_j = einops.repeat(x, 'b r c d -> b x r c d', x=num_rows)

            row, col = torch.triu_indices(num_rows, num_rows, offset=1)

            x_i = x_i[:, row, col]
            x_j = x_j[:, row, col]

            scores = self.decode_gg(x_i, x_j, seq_mask, ij_indices=(row, col))


        elif logits_prev is not None:

            actions_i_prev = actions_ij_prev[:, 0]

            new_scores_indices = torch.cartesian_prod(actions_i_prev, torch.arange(num_rows, device=batch_input.device, dtype=torch.int32)).to(torch.int64)
            new_scores_indices = einops.rearrange(new_scores_indices, '(b n) k -> b n k', n=num_rows)
            new_scores_indices,_ = torch.sort(new_scores_indices, axis=-1)
            
            brange = torch.arange(batch_size).unsqueeze(1).to(batch_input.device)

            x_i = batch_input[brange, new_scores_indices[:,:,0], :, :]
            x_j = batch_input[brange, new_scores_indices[:,:,1], :, :]

            new_scores = self.decode_gg(x_i, x_j, seq_mask, ij_indices=(new_scores_indices[:,:,0], new_scores_indices[:,:,1]))

            logits_prev_new = torch.cat([logits_prev, new_scores], dim=-1)

            scores = torch.gather(logits_prev_new, 1, score_indices_to_prev)


        ret = {
            'logits': scores,
            'distance': scores
        }

        return ret
