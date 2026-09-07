"""
Demonstration of Elastic Weight Consolidation EWC Skill
"""

from client import EWCRegularizer

def main():
    print("=== Continual Learning with Elastic Weight Consolidation (EWC) ===")
    ewc = EWCRegularizer(lambda_ewc=50.0)

    # Task A: Model converged to optimal weights
    task_a_weights = {"w_layer1": 1.5, "w_layer2": -0.8, "b_bias": 0.2}
    task_a_gradients = [
        {"w_layer1": 0.8, "w_layer2": 0.1, "b_bias": 0.05},
        {"w_layer1": 0.9, "w_layer2": 0.05, "b_bias": 0.02},
        {"w_layer1": 0.75, "w_layer2": 0.15, "b_bias": 0.04}
    ]

    print("Registering completed Task A...")
    ewc.register_completed_task(task_a_weights, task_a_gradients)
    print("Task A Fisher Diagonal:", ewc.consolidated_tasks[0][1])

    # Task B: Weights deviate as new task trains
    task_b_weights_candidate = {"w_layer1": 1.2, "w_layer2": -0.5, "b_bias": 0.3}
    penalty, grads = ewc.compute_penalty_and_gradient(task_b_weights_candidate)

    print(f"\nEvaluating Task B Deviation:")
    print(f"  EWC Forgetting Penalty: {penalty:.4f}")
    print(f"  Penalty Gradients: {grads}")

    assert penalty > 0.0
    print("\nElastic Weight Consolidation Verification PASS!")

if __name__ == "__main__":
    main()
