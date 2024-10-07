
def _adjacent_merge(x: torch.Tensor, 
           unm_idx,
           src_idx,
           dst_idx,
           r,
           mode="mean") -> torch.Tensor:
    
    src, dst = x[:], x[:]
    n, t1, c = src.shape
    unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
    src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
    dst = src.gather(dim=-2, index=dst_idx.expand(n, r, c))
    dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

    return torch.cat([unm, dst], dim=1)


# import torch
from torch.nn.utils.rnn import pad_sequence

# def merge_consecutive_trues_batch(x_batch, mask_batch):
#     merged_batches = []
    
#     for x, mask in zip(x_batch, mask_batch):
#         if len(x) == 0:
#             merged_batches.append(x)
#             continue
        
#         mask = mask.bool()
        
#         # 마스크 변화 지점 찾기
#         mask_diff = torch.diff(mask.int())
#         mask_diff = torch.cat((torch.tensor([0], device=mask.device), mask_diff))
        
#         # 그룹 ID 할당
#         group_ids = torch.cumsum(mask_diff != 0, dim=0)
        
#         # 고유한 그룹 ID 추출
#         unique_groups = torch.unique(group_ids)
        
#         merged = []
#         for group in unique_groups:
#             indices = (group_ids == group).nonzero(as_tuple=True)[0]
#             if mask[indices[0]]:
#                 # 마스크가 True인 그룹은 합산
#                 sum_val = x[indices].sum()
#                 merged.append(sum_val.item())
#             else:
#                 # 마스크가 False인 그룹은 개별 값 유지
#                 merged.extend(x[indices].tolist())
        
#         merged_batches.append(torch.tensor(merged))
    
#     merged_padded = pad_sequence(merged_batches, batch_first=True, padding_value=0)
    
#     return merged_padded


import torch

# 예제 데이터
x = torch.tensor([
    [0, 1, 2, 4, 7, 5, 3, 1, 9, 10],
    [3, 2, 1, 0, 4, 4, 4, 2, 2, 2]
])

merge_mask = torch.tensor([
    [True, True, True, False, True, True, False, False, False, False],
    [True, False, True, True, False, True, True, True, False, False]
])

def find_mask_changes(mask_batch):
    # mask_batch: (batch_size, seq_len)
    # Convert mask to int for difference operation
    mask_int = mask_batch.int()
    # Compute difference along the sequence dimension
    mask_diff = torch.diff(mask_int, dim=1)
    # Prepend a zero to maintain the original sequence length
    mask_diff = torch.cat([torch.zeros((mask_diff.size(0), 1), dtype=mask_diff.dtype, device=mask_diff.device), mask_diff], dim=1)
    # A change occurs where mask_diff != 0
    mask_change = mask_diff != 0
    return mask_change

def assign_group_ids(mask_change):
    # mask_change: (batch_size, seq_len) - True where changes occur
    # Compute cumulative sum of changes to assign unique group IDs
    group_ids = torch.cumsum(mask_change, dim=1)
    return group_ids

def merge_consecutive_trues_batch(x_batch, mask_batch):
    batch_size, seq_len = x_batch.size()
    
    # Step 1: Find mask changes
    mask_change = find_mask_changes(mask_batch)
    
    # Step 2: Assign group IDs
    group_ids = assign_group_ids(mask_change)
    
    # To ensure group IDs are unique across the batch, offset them by batch index
    # First, find the maximum group ID per batch
    max_group_ids = torch.max(group_ids, dim=1, keepdim=True).values
    # Compute offsets
    # The maximum possible number of groups per batch is max_group_ids
    max_groups = torch.max(max_group_ids).item()
    offsets = torch.arange(batch_size, device=x_batch.device).unsqueeze(1) * (max_groups + 1)
    # Apply offsets to group_ids
    unique_group_ids = group_ids + offsets
    
    # Flatten the batch for grouping
    flat_x = x_batch.view(-1)
    flat_mask = mask_batch.view(-1)
    flat_group_ids = unique_group_ids.view(-1)
    
    # Get unique group IDs and inverse indices
    unique_groups, inverse_indices = torch.unique(flat_group_ids, sorted=True, return_inverse=True)
    
    # Initialize a tensor to hold the summed values for True groups
    summed = torch.zeros_like(unique_groups, dtype=x_batch.dtype)
    # Compute sums where mask is True
    summed += torch.scatter_add(torch.zeros_like(summed), 0, inverse_indices, flat_x * flat_mask.int())
    
    # Determine if each group is a True group (if the first element in the group is True)
    # To find the first element in each group, we can use scatter
    # Create a tensor indicating the first occurrence of each group
    group_first_mask = torch.zeros_like(unique_groups, dtype=torch.bool)
    # Find the first occurrence by checking where the group ID changes
    # This is already handled by unique, so we can take the first mask value per group
    # Using scatter to set the first occurrence
    first_masks = torch.scatter(flat_mask, 0, inverse_indices, flat_mask).bool()
    group_first_mask = first_masks
    
    # Initialize a list to hold the results for each batch
    merged_batches = [[] for _ in range(batch_size)]
    
    # Iterate over each unique group
    for i, group in enumerate(unique_groups):
        # Find which batch this group belongs to
        batch_idx = group // (max_groups + 1)
        if group_first_mask[i]:
            # If the group is True, append the summed value
            merged_batches[batch_idx].append(summed[i].item())
        else:
            # If the group is False, append individual elements
            # Find the elements belonging to this group
            elements = flat_x[inverse_indices == i]
            merged_batches[batch_idx].extend(elements.tolist())
    
    # Convert merged_batches to a list of tensors
    merged_batches = [torch.tensor(batch) for batch in merged_batches]
    
    merged_padded = pad_sequence(merged_batches, batch_first=True, padding_value=0)
    
    return merged_padded



def adjacent_matching(
    x: torch.Tensor,
    metric: torch.Tensor,
    r: int,
    size: torch.Tensor,
    pos: torch.Tensor,
    key_padding_mask: torch.Tensor,
):

    # We can only reduce by a maximum of 50% tokens
    bs, seq_len, dim = metric.shape
    r = min(r, seq_len // 2)

    if r <= 0:
        return do_nothing(x), do_nothing(size), do_nothing(pos), do_nothing(key_padding_mask)

    with torch.no_grad():
        a = metric[..., :, :]
        b = torch.zeros_like(a)
        b[..., :-1, :] = metric[..., 1:, :]
        scores = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    
        # TODO : need key padding mask ?
        pad_idx = torch.argmax(key_padding_mask.float(), dim=1).squeeze(-1) - 1
        scores[torch.arange(bs), pad_idx] = -math.inf
        scores[:, -1] = -math.inf

        edge_max, edge_idx = torch.sort(scores, dim=1, descending=True)
        unm_idx = edge_idx[..., r:, None]  # Unmerged Tokens
        src_idx = edge_idx[..., :r:, None]  # Merged Tokens
        dst_idx = src_idx + 1

        if size is None:
            size = torch.ones_like(x[..., 0, None])

        x = _adjacent_merge(x * size, unm_idx, src_idx, dst_idx, r, mode="sum")
        size = _adjacent_merge(size, unm_idx, src_idx, dst_idx, r, mode="sum")
        pos = _adjacent_merge(pos, unm_idx, src_idx, dst_idx, r, mode="sum")
        key_padding_mask = _adjacent_merge(key_padding_mask, unm_idx, src_idx, dst_idx, r, mode="sum")

        x = x / size
        return x, size, pos, key_padding_mask