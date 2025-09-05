# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


from typing import Tuple

import torch
import torch.nn.functional as F
from hamer.models.backbones.vit import Block, Attention, ViT

from tmu.merge import bipartite_soft_matching, merge_source, merge_wavg
from tmu.utils import parse_r


class TMUBlock(Block):
    """
    Modifications:
     - Apply TMU between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        attn_size = self._tmu_info["size"] if self._tmu_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)

        r = self._tmu_info["r"].pop(0)
        if r > 0:
            # Apply TMU here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tmu_info["class_token"],
                self._tmu_info["distill_token"],
            )
            if self._tmu_info["trace_source"]:
                self._tmu_info["source"] = merge_source(
                    merge, x, self._tmu_info["source"]
                )
            x, self._tmu_info["size"] = merge_wavg(merge, x, self._tmu_info["size"])

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


class TMUAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape

        # modify to use F.scaled_dot_product_attention
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        k_mean = k.mean(dim=1)

        attn_mask = None
        if size is not None:
            # The mask needs to be broadcastable to the attention matrix shape (B, num_heads, N, N)
            attn_mask = size.log()[:, None, None, :, 0]

        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, k_mean


def make_tmu_class(transformer_class):
    class TMUVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        - Overwrites the forward method to handle token merging and unmerging.
        """
        def _prune_forward(self, x):
            B, C, H, W = x.shape
            x, (Hp, Wp) = self.patch_embed(x)

            if self.pos_embed is not None:
                # fit for multiple GPU training
                # since the first element for pos embed (sin-cos manner) is zero, it will cause no difference
                x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

            for blk in self.blocks:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
            
            # print(f"\nx shape after blocks: {x.shape}")
            # print(f"Number of tokens left: {x.shape[1]}")

            source_matrix = None
            if hasattr(self, "_tmu_info") and "source" in self._tmu_info:
                source_matrix = self._tmu_info["source"]
                # print("source_matrix shape:", source_matrix.shape)
            
            if source_matrix is None:
                B, N, C = x.shape
                print("\nWarning: TMU routing matrix 'r' not found. Unmerge will have no effect.")
                source_matrix = torch.eye(N, N, device=x.device, dtype=x.dtype).expand(B, N, N)
            
            # Apply the unmerge function
            unmerged_features = self.unmerge_tmu_output(x, source_matrix)
            # print("unmerged_features shape:", unmerged_features.shape)

            x = self.last_norm(unmerged_features)
            # print(f"x shape after last norm: {x.shape}")
            return x

        def unmerge_tmu_output(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
            """
            Unmerges the output of a TMU-applied layer back to the original token size.
            """
            source_transposed = source.transpose(1, 2)
            sparse_source = source_transposed.to_sparse()
            # (B, Original_N, Merged_N) @ (B, Merged_N, F) -> (B, Original_N, F)
            unmerged_x = torch.bmm(sparse_source, x)
            return unmerged_x

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tmu_info["r"] = parse_r(len(self.blocks), self.r)
            self._tmu_info["size"] = None
            self._tmu_info["source"] = None
            
            # Call the original forward logic (which now includes unmerging)
            return self._prune_forward(*args, **kwdargs)

    return TMUVisionTransformer


def apply_patch(
    model: ViT, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies TMU to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tmu_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    TMUVisionTransformer = make_tmu_class(model.__class__)

    model.__class__ = TMUVisionTransformer
    model.r = 0
    model._tmu_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": False,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tmu_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = TMUBlock
            module._tmu_info = model._tmu_info
        elif isinstance(module, Attention):
            module.__class__ = TMUAttention
