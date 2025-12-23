import re
from typing import List, Dict, Optional, Any, Tuple
from datasets import load_dataset

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def extract_gsm8k_answer(text: str) -> Optional[str]:
   """
   Extracts the value from the last <answer>...</answer> tag in the text.
   """
   parts = text.split("<answer>")
   if len(parts) < 2: 
       return None
   last_part = parts[-1]
   if "</answer>" not in last_part:
       return None
   answer = last_part.split("</answer>")[0].strip()
   return None if answer == "..." else answer


def extract_gsm8k_answer_from_dataset(text: str) -> Optional[str]:
   """
   Extracts the answer from the GSM8K dataset examples.
   """
   if "####" not in text:
       return None
   return text.split("####")[1].strip()


def build_prompt(messages: List[Dict[str, str]]) -> str:
   """Combines messages into a single prompt string."""
   return "\n".join([msg["content"].strip() for msg in messages])


def prepare_dataset(split: str = "train") -> List[Dict[str, Optional[str]]]:
   """Load and prepare the GSM8K dataset for training."""
   try:
       data = load_dataset('openai/gsm8k', 'main')[split]
   except Exception:
       # Fallback or allow error to propagate if dataset access issue
       print("Warning: Could not load openai/gsm8k dataset automatically. Ensure you have access or internet connection.")
       return []
       
   formatted_data = []
   for example in data:
       prompt_str = build_prompt([
           {"role": "system", "content": SYSTEM_PROMPT},
           {"role": "user", "content": example["question"]}
       ])
       formatted_example = {
           "prompt": prompt_str,  
           "answer": extract_gsm8k_answer_from_dataset(example["answer"])
       }
       formatted_data.append(formatted_example)
   return formatted_data


def extract_last_number(text: str) -> Optional[float]:
   text = text.replace('$', '').replace('%', '')
   pattern = r'(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$'
   match = re.search(pattern, text)
   return float(match.group(1)) if match else None


def extract_single_number(text: str) -> Optional[float]:
   numbers = re.findall(r'-?\d*\.?\d+', text)
   return float(numbers[0]) if len(numbers) == 1 else None


def gsm8k_metric(predicted: str, expected: str) -> Tuple[bool, float]:
    if predicted == expected:  # Exact match
        return True, 2.0
        
    pred_num = extract_single_number(str(predicted))
    exp_num = extract_single_number(str(expected))
    
    if pred_num is not None and exp_num is not None and pred_num == exp_num:
        return True, 1.5
    
    pred_num = extract_last_number(str(predicted))
    exp_num = extract_last_number(str(expected))
    
    if pred_num is not None and exp_num is not None and pred_num == exp_num:
         return True, 0.0 
    
    return False, 0.0


def functional_reward_fn(completions: List[List[Dict[str, str]]], answers: List[str]) -> List[float]: 
    responses = [completion[0]['content'] for completion in completions]
    extracted = [extract_gsm8k_answer(response) for response in responses]
    rewards = [] 
    for pred, exp in zip(extracted, answers):
        if pred is None:
            rewards.append(0.0)
            continue
        _, reward = gsm8k_metric(pred, exp)
        rewards.append(reward)
    return rewards


def structural_reward_fn(completions: List[List[Dict[str, str]]]) -> List[float]: 
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for response in responses:
        score = 0.0
        if "<reasoning>" in response: score += 0.2
        if "</reasoning>" in response: score += 0.2
        if "<answer>" in response: score += 0.2
        if "</answer>" in response: score += 0.2
        rewards.append(score)
    return rewards


def reward_fn(completions: List[List[Dict[str, str]]], answers: List[str]) -> List[float]: 
    functional_rewards = functional_reward_fn(completions, answers)
    structural_rewards = structural_reward_fn(completions)

    combined_rewards = []
    for f_score, s_score in zip(functional_rewards, structural_rewards):
        combined_rewards.append(f_score + s_score)

    return combined_rewards