import torch
import torch.nn as nn

from timm.models.vision_transformer import Attention, Block, VisionTransformer
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py

from typing import Tuple
import math

# https://github.com/facebookresearch/ToMe/blob/main/tome/patch/timm.py
class ToMeSelfAttention(nn.Module):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None
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

        # Key padding mask
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x, k.mean(1)

class ToMeCrossAttention(Attention):
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

# # https://github.com/facebookresearch/ToMe/blob/main/tome/patch/timm.py
# class ToMeBlock(Block):
#     """
#     Modifications:
#      - Apply ToMe between the attention and mlp blocks
#      - Compute and propogate token size and potentially the token sources.
#     """

#     def _drop_path1(self, x):
#         return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

#     def _drop_path2(self, x):
#         return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Note: this is copied from timm.models.vision_transformer.Block with modifications.
#         attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
#         x_attn, metric = self.attn(self.norm1(x), attn_size) # ToMeAttention
#         x = x + self._drop_path1(x_attn)

#         r = self._tome_info["r"].pop(0)
#         if r > 0:
#             # Apply ToMe here
#             merge, _ = bipartite_soft_matching(
#                 metric,
#                 r,
#                 self._tome_info["class_token"],
#                 self._tome_info["distill_token"],
#             )
#             if self._tome_info["trace_source"]:
#                 self._tome_info["source"] = merge_source(
#                     merge, x, self._tome_info["source"]
#                 )
#             x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

#         x = x + self._drop_path2(self.mlp(self.norm2(x)))
#         return x

# # https://github.com/facebookresearch/ToMe/blob/main/tome/patch/mae.py#L22
# class ToMeVisionTransformer(transformer_class):
#     """
#     Modifications:
#     - Initialize r, token size, and token sources.
#     - For MAE: make global average pooling proportional to token size
#     """

#     def forward(self, *args, **kwdargs) -> torch.Tensor:
#         self._tome_info["r"] = parse_r(len(self.blocks), self.r)
#         self._tome_info["size"] = None
#         self._tome_info["source"] = None

#         return super().forward(*args, **kwdargs)

#     def forward_features(self, x: torch.Tensor) -> torch.Tensor:
#         # From the MAE implementation
#         B = x.shape[0]
#         x = self.patch_embed(x)

#         T = x.shape[1]

#         cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = x + self.pos_embed
#         x = self.pos_drop(x)

#         for blk in self.blocks:
#             x = blk(x) # ToMeBlock

#         if self.global_pool:
#             # ---- ToMe changes this ----
#             # Global average pool proportional to token size
#             if self._tome_info["size"] is not None:
#                 x = (x * self._tome_info["size"])[:, 1:, :].sum(dim=1) / T
#             else:
#                 x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
#             # ---- End of change ----

#             outcome = self.fc_norm(x)
#         else:
#             x = self.norm(x)
#             outcome = x[:, 0]

#         return outcome

# return ToMeVisionTransformer

def do_nothing(x, mode=None):
    return x


def _adjacent_merge(x: torch.Tensor, 
           unm_idx,
           src_idx,
           dst_idx,
           sorted_indices,
           r,
           mode="mean") -> torch.Tensor:
    
    t = x.shape[1]
    src, dst = x[..., ::2, :], x[..., 1::2, :]
    n, t1, c = src.shape

    if x.dtype != torch.bool:
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        
        com_val = torch.cat([unm, dst], dim=1)
        sorted_values = torch.gather(com_val, dim=1, index=sorted_indices.unsqueeze(-1).expand(n, t - r, c))
        
        return sorted_values
    
    else:
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c)).float()
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c)).float()
        dst = dst.float().scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        
        com_val = torch.cat([unm, dst], dim=1)
        sorted_values = torch.gather(com_val, dim=1, index=sorted_indices.unsqueeze(-1).expand(n, t - r, c))
        
        return sorted_values.squeeze(-1).bool()

def adjacent_matching(
    x: torch.Tensor,
    metric: torch.Tensor,
    r: int,
    size: torch.Tensor,
    pos: torch.Tensor,
    key_padding_mask: torch.Tensor,
):
    # We can only reduce by a maximum of 50% tokens
    n, t, _ = metric.shape
    r = min(r, t // 2)
    order = torch.arange(t, device='cuda').expand(n, t)
    
    if r <= 0:
        return do_nothing(x), do_nothing(size), do_nothing(pos), do_nothing(key_padding_mask)

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        pad_a, pad_b = key_padding_mask[..., ::2, :], key_padding_mask[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        # TODO : need key padding mask ?
        # 두 마스크를 OR 연산하여 최종 마스크 생성
        pad_mask = pad_a | pad_b.transpose(1,2)  # shape: (38, 37)

        rows = torch.arange(a.size(1), device='cuda').view(-1, 1)
        cols = torch.arange(b.size(1), device='cuda').view(1, -1)
        adjacent_mask = ~((rows == cols) | (rows == cols + 1))
    
        mask = pad_mask | adjacent_mask.unsqueeze(0)
        scores = scores.masked_fill(mask, float('-inf'))
        
        
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        src_order, dst_order = order[..., ::2], order[..., 1::2]
        unm_order = src_order.gather(1, unm_idx.squeeze(-1))
        # src_order = src_order.gather(1, src_idx.squeeze(-1))
        # dst_order = dst_order.gather(1, dst_idx.squeeze(-1))
        com_order = torch.cat([unm_order, dst_order], dim=1)
        _, sorted_indices = com_order.sort(dim=1)
        
        if size is None:
            size = torch.ones_like(x[..., 0, None])

        x = _adjacent_merge(x * size, unm_idx, src_idx, dst_idx, sorted_indices, r, mode="sum")
        pos = _adjacent_merge(pos * size, unm_idx, src_idx, dst_idx, sorted_indices, r, mode="sum")
        size = _adjacent_merge(size, unm_idx, src_idx, dst_idx, sorted_indices, r, mode="sum")
        key_padding_mask = _adjacent_merge(key_padding_mask, unm_idx, src_idx, dst_idx, sorted_indices, r, mode="sum")

        x = x / size
        pos = pos / size
        return x, size, pos, key_padding_mask

def _bipartite_merge(x: torch.Tensor, 
           unm_idx,
           src_idx,
           dst_idx,
           r,
           mode="mean") -> torch.Tensor:
    
    src, dst = x[..., ::2, :], x[..., 1::2, :]
    n, t1, c = src.shape

    if x.dtype != torch.bool:
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        return torch.cat([unm, dst], dim=1) 
    
    else:
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c)).float()
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c)).float()
        dst = dst.float().scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce='mean')
        return torch.cat([unm, dst], dim=1).squeeze(-1).bool()

def bipartite_matching(
    x: torch.Tensor,
    metric: torch.Tensor,
    r: int,
    size: torch.Tensor,
    pos: torch.Tensor,
    key_padding_mask: torch.Tensor,
):

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, t // 2)

    if r <= 0:
        return do_nothing(x), do_nothing(size), do_nothing(pos), do_nothing(key_padding_mask)

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        pad_a, pad_b = key_padding_mask[..., ::2, :], key_padding_mask[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        # TODO : need key padding mask ?
        # 두 마스크를 OR 연산하여 최종 마스크 생성
        pad_mask = pad_a | pad_b.transpose(1,2)  # shape: (38, 37)
        scores = scores.masked_fill(pad_mask, float('-inf'))

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if size is None:
            size = torch.ones_like(x[..., 0, None])

        x = _bipartite_merge(x * size, unm_idx, src_idx, dst_idx, r, mode="sum")
        size = _bipartite_merge(size, unm_idx, src_idx, dst_idx, r, mode="sum")
        pos = _bipartite_merge(pos, unm_idx, src_idx, dst_idx, r, mode="sum")
        key_padding_mask = _bipartite_merge(key_padding_mask, unm_idx, src_idx, dst_idx, r, mode="sum")

        x = x / size
        return x, size, pos, key_padding_mask
    
# def bipartite_soft_matching(
#     metric: torch.Tensor,s
#     r: int,
#     class_token: bool = False,
#     distill_token: bool = False,
# ) -> Tuple[Callable, Callable]:
#     """
#     Applies ToMe with a balanced matching set (50%, 50%).

#     Input size is [batch, tokens, channels].
#     r indicates the number of tokens to remove (max 50% of tokens).

#     Extra args:
#      - class_token: Whether or not there's a class token.
#      - distill_token: Whether or not there's also a distillation token.

#     When enabled, the class token and distillation tokens won't get merged.
#     """
#     protected = 0
#     if class_token:
#         protected += 1
#     if distill_token:
#         protected += 1

#     # We can only reduce by a maximum of 50% tokens
#     t = metric.shape[1]
#     r = min(r, (t - protected) // 2)

#     if r <= 0:
#         return do_nothing, do_nothing

#     with torch.no_grad():
#         metric = metric / metric.norm(dim=-1, keepdim=True)
#         a, b = metric[..., ::2, :], metric[..., 1::2, :]
#         scores = a @ b.transpose(-1, -2)

#         if class_token:
#             scores[..., 0, :] = -math.inf
#         if distill_token:
#             scores[..., :, 0] = -math.inf

#         node_max, node_idx = scores.max(dim=-1)
#         edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

#         unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
#         src_idx = edge_idx[..., :r, :]  # Merged Tokens
#         dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

#         if class_token:
#             # Sort to ensure the class token is at the start
#             unm_idx = unm_idx.sort(dim=1)[0]

#     def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
#         src, dst = x[..., ::2, :], x[..., 1::2, :]
#         n, t1, c = src.shape
#         unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
#         src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
#         dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

#         if distill_token:
#             return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
#         else:
#             return torch.cat([unm, dst], dim=1)


# def token_merging(
#     merge: Callable, x: torch.Tensor, size: torch.Tensor = None
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Applies the merge function by taking a weighted average based on token size.
#     Returns the merged tensor and the new token sizes.
#     """
#     if size is None:
#         size = torch.ones_like(x[..., 0, None])

#     x = merge(x * size, mode="sum")
#     size = merge(size, mode="sum")

#     x = x / size
#     return x, size
