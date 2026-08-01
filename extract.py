import json
with open('analysis/AQI_Analysis.ipynb', 'r') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        if 'outputs' in cell:
            for out in cell['outputs']:
                if 'text' in out:
                    print(out['text'])
