import json
path = 'notebooks/curvature_bifurcation.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        if 'outputs' not in cell:
            cell['outputs'] = []
        if 'execution_count' not in cell:
             cell['execution_count'] = None
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook fixed.")
