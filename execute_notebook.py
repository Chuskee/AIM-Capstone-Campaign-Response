"""Execute the revised notebook using this Python environment and save outputs."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
runtime = ROOT / '.runtime'
kernel = runtime / 'kernels' / 'capstone-local'
kernel.mkdir(parents=True, exist_ok=True)
(kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
    'display_name':'Capstone local environment','language':'python'}))
os.environ['JUPYTER_PATH'] = str(runtime)
os.environ['JUPYTER_RUNTIME_DIR'] = str(runtime/'connections')
os.environ['IPYTHONDIR'] = str(runtime/'ipython')
os.environ['MPLCONFIGDIR'] = str(runtime/'matplotlib')
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key] = '1'
os.chdir(ROOT)
import nbformat
from nbclient import NotebookClient
path=ROOT/'notebooks/Capstone_Customer_Segmentation_Revised.ipynb'
nb=nbformat.read(path,as_version=4)
def started(cell,cell_index,**kwargs):
    if cell.cell_type=='code': print(f'Executing cell {cell_index}: {cell.source.splitlines()[0][:100]}',flush=True)
def completed(cell,cell_index,**kwargs):
    for output in cell.get('outputs',[]):
        if output.get('output_type')=='stream': print(output.get('text','')[-1800:],flush=True)
client=NotebookClient(nb,timeout=1800,kernel_name='capstone-local',resources={'metadata':{'path':str(ROOT)}},
    on_cell_start=started,on_cell_complete=completed)
try:
    client.execute()
finally:
    nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
    nbformat.write(nb,path)
print(f'Executed notebook saved: {path}',flush=True)
