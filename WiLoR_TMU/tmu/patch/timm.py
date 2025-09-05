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
# from timm.models.vision_transformer import Attention, Block, VisionTransformer
from wilor.models.backbones.vit import Block, Attention, ViT
from wilor.utils.geometry import rot6d_to_rotmat, aa_to_rotmat

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
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x, k.mean(1)


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
            # X [B, 192, 1280]
            # x cat [ mean_pose, mean_shape, mean_cam] tokens 
            pose_tokens  = self.pose_emb(self.init_hand_pose.reshape(1, self.cfg.MANO.NUM_HAND_JOINTS + 1, self.joint_rep_dim)).repeat(B, 1, 1)
            shape_tokens = self.shape_emb(self.init_betas).unsqueeze(1).repeat(B, 1, 1)
            cam_tokens   = self.cam_emb(self.init_cam).unsqueeze(1).repeat(B, 1, 1)
            
            x = torch.cat([pose_tokens, shape_tokens, cam_tokens, x], 1)
            for blk in self.blocks:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)

            # print(f"\nx shape after blocks: {x.shape}")
            # print(f"Number of tokens left: {x.shape[1]-18}")

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

            pose_feat  = x[:, :(self.cfg.MANO.NUM_HAND_JOINTS + 1)]
            shape_feat = x[:, (self.cfg.MANO.NUM_HAND_JOINTS + 1):1+(self.cfg.MANO.NUM_HAND_JOINTS + 1)]
            cam_feat   = x[:, 1+(self.cfg.MANO.NUM_HAND_JOINTS + 1):2+(self.cfg.MANO.NUM_HAND_JOINTS + 1)]
            
            #print(pose_feat.shape, shape_feat.shape, cam_feat.shape)
            pred_hand_pose = self.decpose(pose_feat).reshape(B, -1) + self.init_hand_pose  #B , 96
            pred_betas = self.decshape(shape_feat).reshape(B, -1) + self.init_betas        #B , 10
            pred_cam = self.deccam(cam_feat).reshape(B, -1) + self.init_cam                #B , 3
            
            pred_mano_feats = {}
            pred_mano_feats['hand_pose'] = pred_hand_pose
            pred_mano_feats['betas']     = pred_betas
            pred_mano_feats['cam']       = pred_cam

            
            joint_conversion_fn = {
                    '6d': rot6d_to_rotmat,
                    'aa': lambda x: aa_to_rotmat(x.view(-1, 3).contiguous())
                }[self.joint_rep_type]
    
            pred_hand_pose = joint_conversion_fn(pred_hand_pose).view(B, self.cfg.MANO.NUM_HAND_JOINTS+1, 3, 3)
            pred_mano_params = {'global_orient': pred_hand_pose[:, [0]],
                                'hand_pose': pred_hand_pose[:, 1:],
                                'betas': pred_betas}
            
            img_feat = x[:, 2+(self.cfg.MANO.NUM_HAND_JOINTS + 1):].reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2)
            # print("\nShape of img_feat:", img_feat.shape)
            return pred_mano_params, pred_cam, pred_mano_feats, img_feat


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
