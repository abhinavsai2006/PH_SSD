import json, sys

with open('PH_SSD_Colab_Standalone.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        code.append(''.join(cell['source']))

full_code = "import sys, os\ntry:\n    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)\n    sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)\nexcept Exception:\n    pass\n\n" + '\n\n# --- CELL SEPARATOR ---\n\n'.join(code)

with open('run_standalone.py', 'w', encoding='utf-8') as f:
    f.write(full_code)

print('Generated run_standalone.py with line_buffering=True.')
