"""
MCP Server for Elastic Weight Consolidation EWC Skill
"""

import json
import sys
from client import EWCRegularizer

ewc = EWCRegularizer()

def handle_call(name: str, args: dict) -> dict:
    if name == "register_task":
        w = args.get("weights", {})
        g = args.get("sample_gradients", [])
        ewc.register_completed_task(w, g)
        return {"status": "task_registered", "total_tasks": len(ewc.consolidated_tasks)}
    elif name == "compute_penalty":
        w = args.get("weights", {})
        p, g = ewc.compute_penalty_and_gradient(w)
        return {"ewc_penalty": p, "penalty_gradients": g}
    return {"error": f"Unknown tool: {name}"}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        res = handle_call(req.get("method"), req.get("params", {}))
        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
