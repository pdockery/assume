"""Dump cell outlines of both teaching notebooks + shared-cell diff."""
import json, sys

TB = r"C:\Users\docke\Projects\ASSUME\assume\testbed"

def load(name):
    with open(f"{TB}\\{name}", encoding="utf-8") as f:
        return json.load(f)

def outline(nb, label):
    lines = [f"=== {label} ({len(nb['cells'])} cells) ==="]
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        first = next((l for l in src.splitlines() if l.strip()), "")[:110]
        lines.append(f"{i:3d} {c.get('id','?'):28s} {c['cell_type'][:4]:4s} | {first}")
    return "\n".join(lines)

cra = load("congestion_revenue_allocation_example.ipynb")
sea = load("seams_examples.ipynb")

out = [outline(cra, "CRA"), "", outline(sea, "SEAMS"), ""]

# shared-id check
cra_by_id = {c.get("id"): c for c in cra["cells"]}
sea_by_id = {c.get("id"): c for c in sea["cells"]}
shared = [i for i in cra_by_id if i in sea_by_id]
out.append(f"=== SHARED IDS ({len(shared)}) ===")
for i in shared:
    same = "".join(cra_by_id[i]["source"]) == "".join(sea_by_id[i]["source"])
    out.append(f"{i:28s} {'IDENTICAL' if same else '** DIFFERS **'}")

with open(f"{TB}\\_nb_outline.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("wrote _nb_outline.txt")
