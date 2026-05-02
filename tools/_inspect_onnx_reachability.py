#!/usr/bin/env python3
"""
Diagnostic: for each graph input, trace forward through node consumers and
report whether the input is reachable to any graph output.

If an input shows "reachable to output: NO", the input is declared but
disconnected -- the engine has the binding but its data has no effect on
the output. That's the failure mode we hit when ControlNet residuals or
controlnet_strength get folded out during export.
"""

import sys
from pathlib import Path

import onnx


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: _inspect_onnx_reachability.py <onnx-path>")
        return 2
    path = Path(sys.argv[1])
    print(f"loading {path}")
    model = onnx.load(str(path), load_external_data=False)
    g = model.graph

    input_names = [i.name for i in g.input]
    output_names = [o.name for o in g.output]
    print(f"graph inputs:  {input_names}")
    print(f"graph outputs: {output_names}")

    # Build tensor->[consumer node] map.
    consumers: dict[str, list] = {}
    for n in g.node:
        for inp in n.input:
            consumers.setdefault(inp, []).append(n)

    # Forward reach from a starting tensor: BFS through consumer nodes.
    def forward_reach(start: str) -> tuple[int, set[str]]:
        seen_nodes: set[str] = set()
        produced: set[str] = set()
        frontier = [start]
        while frontier:
            t = frontier.pop()
            for n in consumers.get(t, ()):
                if n.name in seen_nodes:
                    continue
                seen_nodes.add(n.name)
                for out in n.output:
                    if out in produced:
                        continue
                    produced.add(out)
                    frontier.append(out)
        return len(seen_nodes), produced

    print()
    for name in input_names:
        n_nodes, produced_tensors = forward_reach(name)
        reachable_outputs = [o for o in output_names if o in produced_tensors]
        verdict = "YES" if reachable_outputs else "NO  <-- DISCONNECTED"
        print(f"  {name}:")
        print(f"    forward reach: {n_nodes} ops, {len(produced_tensors)} tensors")
        print(f"    reachable to output: {verdict} {reachable_outputs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
