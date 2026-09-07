"""
Elastic Weight Consolidation EWC Regularization Skill Client
Pure Python Standard Library implementation of Elastic Weight Consolidation (Kirkpatrick et al., DeepMind).
Computes diagonal Fisher Information Matrix (FIM) importance scores for model weights,
penalizing significant deviations from past task optima to prevent catastrophic forgetting.
"""

from typing import List, Dict, Any, Tuple, Optional
import math


class EWCRegularizer:
    def __init__(self, lambda_ewc: float = 100.0):
        self.lambda_ewc = lambda_ewc
        # Past tasks: list of tuples (optimal_weights, fisher_diagonal)
        self.consolidated_tasks: List[Tuple[Dict[str, float], Dict[str, float]]] = []

    def compute_fisher_diagonal(self, gradients_per_sample: List[Dict[str, float]]) -> Dict[str, float]:
        """Estimate diagonal of Fisher Information Matrix: F_i = E[g_i^2]."""
        if not gradients_per_sample:
            return {}
        n = len(gradients_per_sample)
        fisher = {}
        keys = gradients_per_sample[0].keys()
        for k in keys:
            squared_sum = sum(sample[k] ** 2 for sample in gradients_per_sample)
            fisher[k] = squared_sum / n
        return fisher

    def register_completed_task(self, current_weights: Dict[str, float], gradients_per_sample: List[Dict[str, float]]):
        """Freeze current task optimal weights and compute its Fisher importance."""
        fisher = self.compute_fisher_diagonal(gradients_per_sample)
        self.consolidated_tasks.append((dict(current_weights), fisher))

    def compute_penalty_and_gradient(self, current_weights: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Compute EWC quadratic penalty: (lambda/2) * sum_i F_i * (theta_i - theta_A,i)^2."""
        total_penalty = 0.0
        penalty_grads = {k: 0.0 for k in current_weights}

        for theta_star, fisher in self.consolidated_tasks:
            for k, w in current_weights.items():
                if k in theta_star and k in fisher:
                    diff = w - theta_star[k]
                    fim_val = fisher[k]
                    penalty = 0.5 * self.lambda_ewc * fim_val * (diff ** 2)
                    grad = self.lambda_ewc * fim_val * diff
                    total_penalty += penalty
                    penalty_grads[k] += grad

        return total_penalty, penalty_grads
