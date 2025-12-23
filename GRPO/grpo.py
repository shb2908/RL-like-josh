import copy 
import wandb
import torch 
import random 
import numpy as np 
import torch.nn as nn
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, Union

from loss import grpo_loss

def setup_training_environment(device_name: str, dtype: str = "bfloat16") -> Dict:
    """Set up mixed precision and device context."""
    
    # Set up random seed
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set up mixed precision
    device_type = 'cuda' if 'cuda' in device_name else 'cpu'
    
    ptdtype = {
        'float32': torch.float32, 
        'bfloat16': torch.bfloat16, 
        'float16': torch.float16
    }[dtype]
    
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    return {
        'device': device_name,
        'ctx': ctx,
        'device_type': device_type,
    }

def get_memory_usage() -> str:
    """Get current GPU memory usage in a human-readable format."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)   
        return f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    return "CUDA not available"

def set_random_seed(seed: int = 42):
    """Set the random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def selective_log_softmax(logits: torch.Tensor, input_ids: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
    """
    Compute log probabilities for the given input_ids using the logits, chunking to save memory.
    """
    device = logits.device
    batch_size, seq_len, vocab_size = logits.shape
    log_probs = torch.zeros(batch_size, seq_len, device=device)
    
    for i in range(0, seq_len, chunk_size):
        end_idx = min(i + chunk_size, seq_len)
        chunk_logits = logits[:, i:end_idx, :]
        chunk_ids = input_ids[:, i:end_idx]
        
        chunk_log_probs = nn.functional.log_softmax(chunk_logits, dim=-1)
        
        log_probs[:, i:end_idx] = chunk_log_probs.gather(
            dim=-1, index=chunk_ids.unsqueeze(-1)).squeeze(-1)
        
        del chunk_logits, chunk_log_probs
    
    return log_probs

def compute_log_probs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, logits_to_keep: int, env: Dict, chunk_size: int = 64) -> torch.Tensor: 
    """Compute log probabilities for the last `logits_to_keep` tokens."""
    with env['ctx']: 
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # shape: (B, L-1, V)
    
    target_ids = input_ids[:, 1:]
    
    targets = target_ids[:, -logits_to_keep:]
    current_logits = logits[:, -logits_to_keep:, :]
    
    return selective_log_softmax(current_logits, targets, chunk_size)

def create_completion_mask(completion_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """
    Create a mask for valid completion tokens (excluding padding after EOS).
    """
    is_eos = completion_ids == eos_token_id
    max_len = completion_ids.size(1)
    
    eos_indices = is_eos.int().argmax(dim=1)
    
    has_eos = is_eos.any(dim=1)
    eos_indices = torch.where(has_eos, eos_indices, torch.tensor(max_len, device=completion_ids.device))
    
    sequence_indices = torch.arange(max_len, device=completion_ids.device).expand(completion_ids.size(0), -1)
    
    mask = (sequence_indices < eos_indices.unsqueeze(1)).int()
    
    return mask

def generate_completions(model, tokenizer, prompts: List[str], device, num_generations: int = 4, max_completion_length: int = 32, env=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate multiple completions for each prompt."""
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    prompt_ids = inputs["input_ids"].to(device)
    prompt_mask = inputs["attention_mask"].to(device)
    
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
    
    prompt_length = prompt_ids.size(1)
    
    with env['ctx']:
        outputs = model.generate(
            prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=max_completion_length,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=False
        )
    
    # Extract completion part
    completion_ids = outputs[:, prompt_length:]
    
    # Create mask
    completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id)
    
    return prompt_ids, prompt_mask, completion_ids, completion_mask

def generate_rollout_data(
    model, 
    ref_model, 
    tokenizer, 
    batch_samples, 
    device, 
    num_generations, 
    max_completion_length, 
    env, 
    chunk=64
) -> Dict:
    """
    Performs rollout: generates responses and computes initial log probs.
    """
    prompts = [sample["prompt"] if isinstance(sample, dict) else sample[0] for sample in batch_samples]
    answers = [sample["answer"] if isinstance(sample, dict) else sample[1] for sample in batch_samples]
    
    with torch.no_grad():
        prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(
            model, tokenizer, prompts, device, num_generations, max_completion_length, env
        )
        
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        
        logits_to_keep = completion_ids.size(1)
        
        old_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep, env, chunk)
        ref_log_probs = compute_log_probs(ref_model, input_ids, attention_mask, logits_to_keep, env, chunk)
        
    formatted_completions = [[{'content': tokenizer.decode(ids, skip_special_tokens=True)}] for ids in completion_ids]
    
    repeated_prompts = [p for p in prompts for _ in range(num_generations)]
    repeated_answers = [a for a in answers for _ in range(num_generations)]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "formatted_completions": formatted_completions,
        "repeated_prompts": repeated_prompts,
        "repeated_answers": repeated_answers,
        "logits_to_keep": logits_to_keep,
        "batch_size": len(prompts), 
        "num_generations": num_generations
    }

def compute_advantages(rewards: torch.Tensor, num_generations: int) -> Tuple[torch.Tensor, float]:
    """
    Computes advantageous using group statistics.
    rewards: (batch_size * num_generations, )
    """
    total_batch_size = rewards.size(0)
    num_prompts = total_batch_size // num_generations
    
    rewards_matrix = rewards.view(num_prompts, num_generations)
    
    group_mean = rewards_matrix.mean(dim=1, keepdim=True)
    group_std = rewards_matrix.std(dim=1, keepdim=True)
    
    advantages_matrix = (rewards_matrix - group_mean) / (group_std + 1e-4)
    
    advantages = advantages_matrix.view(-1)
    
    avg_reward = rewards.mean().item()
    
    return advantages, avg_reward

def train_with_grpo(
    model, 
    tokenizer, 
    train_data, 
    num_iterations=1, 
    num_steps=500,
    batch_size=4, 
    num_generations=4, 
    max_completion_length=128,
    beta=0.1, 
    learning_rate=5e-6, 
    mu=3, 
    epsilon=0.2, 
    reward_function=None,
    env=None, 
    gradient_accumulation_steps=1
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if env is None:
        env = setup_training_environment(str(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for iteration in range(num_iterations): 
        print(f"\nIteration {iteration+1}/{num_iterations}")

        ref_model = copy.deepcopy(model)
        ref_model.eval() 
        for param in ref_model.parameters(): 
            param.requires_grad = False 
        print("Reference model created")

        model.train() 

        for step in range(num_steps): 
            print(f"\nStep {step+1}/{num_steps}")
            batch_samples = random.sample(train_data, batch_size)
                
            with torch.no_grad():
                rollout = generate_rollout_data(
                    model, 
                    ref_model, 
                    tokenizer, 
                    batch_samples, 
                    device,
                    num_generations, 
                    max_completion_length, 
                    env
                )
                
                raw_rewards = reward_function(
                    completions=rollout["formatted_completions"], 
                    answers=rollout["repeated_answers"]
                )
                rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float32, device=device)
                
                advantages, avg_reward = compute_advantages(rewards_tensor, num_generations)
                
                print(f"Example response: {rollout['formatted_completions'][0][0]['content']}")
                print(f"Average Reward: {avg_reward:.4f}")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            for grpo_iter in range(mu): 
                new_log_probs = compute_log_probs(
                    model, 
                    rollout["input_ids"], 
                    rollout["attention_mask"], 
                    rollout["logits_to_keep"], 
                    env
                )
                
                loss = grpo_loss(
                    new_log_probs=new_log_probs,
                    old_log_probs=rollout["old_log_probs"],
                    ref_log_probs=rollout["ref_log_probs"],
                    advantages=advantages,
                    mask=rollout["completion_mask"],
                    beta=beta,
                    epsilon=epsilon
                )
                
                optimizer.zero_grad()
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                wandb.log({
                    "loss": loss.item(), 
                    "average_reward": avg_reward, 
                    "step": step + 1,
                    "grpo_iter": grpo_iter + 1,
                })     

                print(f"GRPO iter {grpo_iter+1}/{mu}, loss: {loss.item():.4f}")
                
    return model

def optimize_model_memory(model):
    model.config.use_cache = False
    model = torch.compile(model)
    return model