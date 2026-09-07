# GenPark Elastic Weight Consolidation EWC Skill

Elastic Weight Consolidation (EWC) engine calculating Fisher Information Matrix diagonal penalties to mitigate catastrophic forgetting in continual learning.

Read more at [GenPark](https://genpark.ai) and the [GenPark MCP Catalog](https://genpark.ai/mcp).

```mermaid
graph TD
    A[Task A Loss Minimization] --> B[Task A Optimal Weights theta_A*]
    B --> C[Compute Fisher Diagonal F_i]
    C --> D[Task B Training]
    D --> E[L_total = L_B + (lambda/2) * sum F_i * (theta - theta_A*)^2]
    E --> F[High Performance on Both Task A and Task B]
```

## Features
- Diagonal Fisher Information Matrix (FIM) estimation.
- Quadratic elasticity penalty and analytical gradient calculation.
- Pure Python 3.9+ standard library.
