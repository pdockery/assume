import json
with open('testbed/inspect_rl_results.ipynb') as f:
    nb = json.load(f)
cells = nb['cells']

# Show cells 14-22 to see the full SCED re-run context
for i in range(max(0, 14), min(len(cells), 23)):
    c = cells[i]
    src = ''.join(c['source'])
    ctype = c['cell_type']
    print(f"\n=== Cell {i} id={c.get('id','?')} type={ctype} ===")
    print(src[:600] if len(src) > 600 else src)
    if len(src) > 600:
        print(f"  ... [{len(src)} chars total]")
