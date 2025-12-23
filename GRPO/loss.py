import torch

def grpo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.1,
    epsilon: float = 0.2,
) -> torch.Tensor:
    """
    Computes the GRPO loss function.

    Args:
        new_log_probs: Log probabilities from the current model [batch_size, seq_len]
        old_log_probs: Log probabilities from the old model (during rollout) [batch_size, seq_len]
        ref_log_probs: Log probabilities from the reference model [batch_size, seq_len]
        advantages: Normalized advantages [batch_size, 1] (or broadcastable)
        mask: Mask for completion tokens [batch_size, seq_len]
        beta: KL penalty coefficient
        epsilon: Coloring clip coefficient

    Returns:
        loss: Scalar loss value
    """
    
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)
        
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    surrogate_loss = torch.min(surr1, surr2)
    
    kl_div = torch.exp(ref_log_probs - new_log_probs) - (ref_log_probs - new_log_probs) - 1
    
    per_token_obj = surrogate_loss - beta * kl_div
    
    masked_obj = (per_token_obj * mask).sum(dim=1)
    mask_sum = mask.sum(dim=1)
    
    mask_sum = torch.clamp(mask_sum, min=1.0)
    
    loss = - (masked_obj / mask_sum).mean()
    
    return loss