"""Sub-MoE expert-output clustering primitives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import torch
import torch.nn.functional as F


@dataclass
class GroupingResult:
    labels: torch.Tensor
    iterations: int
    converged: bool
    empty_cluster_events: list[dict[str, int]]


def tokenwise_cosine_similarity(outputs: torch.Tensor, centroid: torch.Tensor, chunk_size: int = 2048,
                                device: str | torch.device = "cuda") -> torch.Tensor:
    """Mean_l cosine(outputs[l], centroid[l]), accumulated in FP32 by chunks."""
    if outputs.ndim != 2 or centroid.shape != outputs.shape:
        raise ValueError("outputs and centroid must both be [tokens, hidden_dim]")
    total = torch.zeros((), device=device, dtype=torch.float32)
    for start in range(0, outputs.shape[0], chunk_size):
        end = min(start + chunk_size, outputs.shape[0])
        total += F.cosine_similarity(outputs[start:end].to(device, torch.float32),
                                     centroid[start:end].to(device, torch.float32), dim=-1).sum()
    return (total / outputs.shape[0]).cpu()


def _similarities(representations: torch.Tensor, centroids: list[torch.Tensor], chunk_size: int,
                  device: str | torch.device) -> torch.Tensor:
    """Vectorized chunked [experts, groups] token-wise cosine means."""
    result = torch.zeros((representations.shape[0], len(centroids)), device=device, dtype=torch.float32)
    for start in range(0, representations.shape[1], chunk_size):
        end = min(start + chunk_size, representations.shape[1])
        rep = F.normalize(representations[:, start:end].to(device, torch.float32), dim=-1)
        cen = F.normalize(torch.stack([c[start:end] for c in centroids]).to(device, torch.float32), dim=-1)
        result += torch.einsum("ecd,kcd->ek", rep, cen)
    return (result / representations.shape[1]).cpu()


def _centroids(representations: torch.Tensor, labels: torch.Tensor, num_groups: int) -> list[torch.Tensor]:
    return [representations[labels == group].float().mean(0).to(torch.bfloat16).cpu()
            for group in range(num_groups)]


def cluster_expert_outputs(representations: torch.Tensor, num_groups: int, *, seed: int = 0,
                           max_iter: int = 100, chunk_size: int = 2048,
                           device: str | torch.device = "cuda") -> GroupingResult:
    """Fixed-K Sub-MoE K-means with token-wise cosine similarity.

    ``representations`` is CPU [experts, tokens, hidden_dim], never a pooled
    [experts, hidden_dim] summary.
    """
    if representations.ndim != 3:
        raise ValueError("representations must be [experts, tokens, hidden_dim]")
    experts = representations.shape[0]
    if not 1 <= num_groups <= experts:
        raise ValueError("num_groups must be in [1, num_experts]")
    reps = representations.detach().to("cpu", torch.bfloat16)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    chosen = [int(torch.randint(experts, (1,), generator=rng))]
    best_similarity = _similarities(reps, [reps[chosen[0]]], chunk_size, device).squeeze(1)
    while len(chosen) < num_groups:
        distances = 1 - best_similarity
        distances[chosen] = 0
        if float(distances.sum()) <= 0:
            candidate = next(index for index in range(experts) if index not in chosen)
        else:
            candidate = int(torch.multinomial(distances.square() / distances.square().sum(), 1, generator=rng))
        chosen.append(candidate)
        best_similarity = torch.maximum(best_similarity, _similarities(reps, [reps[candidate]], chunk_size, device).squeeze(1))
    centroids = [reps[index] for index in chosen]
    labels = torch.full((experts,), -1, dtype=torch.long)
    events: list[dict[str, int]] = []
    for iteration in range(1, max_iter + 1):
        similarities = _similarities(reps, centroids, chunk_size, device)
        new_labels = similarities.argmax(1)
        empty_groups = [group for group in range(num_groups) if not bool((new_labels == group).any())]
        used: set[int] = set()
        for group in empty_groups:
            order = torch.argsort(similarities.gather(1, new_labels[:, None]).squeeze(1))
            chosen_expert = None
            for candidate in order.tolist():
                source = int(new_labels[candidate])
                if candidate not in used and int((new_labels == source).sum()) > 1:
                    chosen_expert = candidate; break
            if chosen_expert is None:
                chosen_expert = next(candidate for candidate in order.tolist() if candidate not in used)
            new_labels[chosen_expert] = group; used.add(chosen_expert)
            events.append({"iteration": iteration, "group": group, "expert": chosen_expert})
        assert all(bool((new_labels == group).any()) for group in range(num_groups))
        unchanged = torch.equal(labels, new_labels)
        labels = new_labels
        centroids = _centroids(reps, labels, num_groups)
        if unchanged:
            return GroupingResult(labels, iteration, True, events)
    return GroupingResult(labels, max_iter, False, events)
