from typing import Tuple, Optional

import torch
import torch.nn.functional as F


def matmul_with_multi_head(
    input: torch.Tensor,
    weight: torch.Tensor,
    num_heads: int = 1,
) -> torch.Tensor:
    """matmul input and weight and return output with multi heads

    Args:
        input (torch.Tensor): input tensor in the range of [-1, 1], with shape: [batch_size, seq_len, hidden_size]
        weight (torch.Tensor): weight tensor in the range of [-1, 1], with shape: [hidden_size, embed_size]
        num_heads (int): number of heads to split hidden_size

    Returns:
        output (torch.Tensor): output tensor, with shape: [batch_size, seqlen, num_heads, embed_size]
    """
    x=input.reshape(input.shape[0],input.shape[1],num_heads,input.shape[2]//num_heads)
    w=weight.reshape(num_heads,weight.shape[0]//num_heads,weight.shape[1])
    out=torch.einsum('bshd,hde->bshe',x,w)
    return out
    


def matmul_with_importance(
    input: torch.Tensor,
    weight: torch.Tensor,
    probs: torch.Tensor,
    grad_output: Optional[torch.Tensor] = None,
    num_heads: int = 1,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """matmul input and weight and return output (with optional grad_input, grad_weight whenever grad_output is given) 
    where only the important elements of the input tensor can be computed and gathered to the output tensor
    decided by the importance probability tensor, tuned by top_p and top_k
    
    Args:
        input (torch.Tensor): input tensor in the range of [-1, 1], with shape: [batch_size, seq_len, hidden_size]
        weight (torch.Tensor): weight tensor in the range of [-1, 1], with shape: [hidden_size, embed_size]
        probs (torch.Tensor): probability tensor in the range of [0, 1], with shape: [batch_size, seq_len]
        grad_output (Optional[torch.Tensor], optional): gradient for the output tensor, with shape: [t, hidden_size]. Defaults to None. Used in Task3.
        num_heads (int): number of heads to split hidden_size
        top_p (float, [0., 1.]): only the elements with the probability equal or higher than top_p are important ones
        top_k (int, [1, ..., seq_len], optional): only the elements with the top_k highest probability are important ones
    
    Returns:
        output (torch.Tensor): output tensor, with shape: [t, num_heads, embed_size]
        grad_input (torch.Tensor, optional): gradient for the input tensor if grad_output is given, otherwise None.
            Return None in Task2.
        grad_weight (torch.Tensor, optional): gradient for the weight tensor if grad_output is given, otherwise None
            Return None in Task2.
    """
    b, s, h = input.shape
    e = weight.shape[1]
    hd = h // num_heads

    # ----- 构建 mask（top_p AND top_k） -----
    mask = probs >= top_p
    if top_k is not None:
        _, topk_idx = torch.topk(probs, top_k, dim=1)
        mask_k = torch.zeros_like(probs, dtype=torch.bool)
        mask_k.scatter_(1, topk_idx, True)
        mask = mask & mask_k

    # ----- 筛选重要位置 -----
    x_sel = input[mask]                                    # [t, h]

    if x_sel.shape[0] == 0:
        out = torch.empty(0, num_heads, e, device=input.device, dtype=input.dtype)
        if grad_output is not None:
            return out, torch.zeros_like(input), torch.zeros_like(weight)
        return out, None, None

    # ----- 多头矩阵乘（同 Task1） -----
    x_heads = x_sel.reshape(-1, num_heads, hd)             # [t, nh, hd]
    w_heads = weight.reshape(num_heads, hd, e)             # [nh, hd, e]
    out = torch.einsum('tnd,nde->tne', x_heads, w_heads)   # [t, nh, e]

    if grad_output is None:
        return out, None, None

    # ----- 手动反向（Task3） -----
    # grad_input: 梯度只能回传到被选中的位置
    grad_x_sel = torch.einsum('tne,nde->tnd', grad_output, w_heads)
    grad_x_sel = grad_x_sel.reshape(x_sel.shape[0], h)
    grad_input = torch.zeros_like(input)
    grad_input[mask] = grad_x_sel

    # grad_weight: 累加所有选中位置对权重的梯度
    grad_w = torch.einsum('tnd,tne->nde', x_heads, grad_output)
    grad_weight = grad_w.reshape(h, e)

    return out, grad_input, grad_weight