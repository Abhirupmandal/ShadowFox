import torch

def apply_temperature(logits, temp):
    return logits / temp

def top_k_filter(logits, k):
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(-1)

    return torch.where(
        logits < min_values,
        torch.tensor(float("-inf")).to(logits.device),
        logits
    )