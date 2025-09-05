# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple, Optional

import torch


def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of image tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # Separate image tokens and MANO tokens
    img_tokens = metric[:, :-18, :]  # All tokens except last 18
    mano_tokens = metric[:, -18:, :]  # Last 18 tokens are MANO tokens
    
    # We can only reduce by a maximum of 50% of image tokens
    t_img = img_tokens.shape[1]
    r = min(r, (t_img - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        # Only work with image tokens for matching
        img_tokens_norm = img_tokens / img_tokens.norm(dim=-1, keepdim=True)
        a, b = img_tokens_norm[..., ::2, :], img_tokens_norm[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        # Separate image tokens and MANO tokens
        img_x = x[:, :-18, :]
        mano_x = x[:, -18:, :]
        
        # Apply merging only to image tokens
        src, dst = img_x[..., ::2, :], img_x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            merged_img = torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            merged_img = torch.cat([unm, dst], dim=1)
        
        # Concatenate merged image tokens with unchanged MANO tokens
        return torch.cat([merged_img, mano_x], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        # Separate merged image tokens and MANO tokens
        merged_img_len = x.shape[1] - 18
        merged_img = x[:, :merged_img_len, :]
        mano_x = x[:, merged_img_len:, :]
        
        unm_len = unm_idx.shape[1]
        unm, dst = merged_img[..., :unm_len, :], merged_img[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        # Reconstruct original image tokens
        out_img = torch.zeros(n, t_img, c, device=x.device, dtype=x.dtype)
        out_img[..., 1::2, :] = dst
        out_img.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out_img.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        # Concatenate with MANO tokens
        return torch.cat([out_img, mano_x], dim=1)

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    merged_x = merge(x * size, mode="sum")
    merged_size = merge(size, mode="sum")

    result_x = merged_x / merged_size
    return result_x, merged_size


def merge_source(
    merge: Callable, x: torch.Tensor, source: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    merged_source = merge(source, mode="amax")
    return merged_source
