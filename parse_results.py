import json
from glob import glob
import os
from tqdm import *
from collections import OrderedDict
import itertools
import sys

def get_hidden_layers():
    x = [128, 256, 512, 768, 1024, 2048]
    hl = []

    for i in range(1, len(x)):
        hl.extend([p for p in itertools.product(x, repeat=i+1)])
    
    return hl

#dirs = glob('train_dir/128-128-*')
dirs = ['-'.join([str(i) for i in hl]) for hl in get_hidden_layers()]
dicts = OrderedDict()
# Load the existing dict
if os.path.isfile('dicts.json'):
    with open('dicts.json', 'r') as f:
        dicts = json.load(f)

for m in tqdm(dirs):
    _file = os.path.join('train_dir', m, 'results.json')
    if m in dicts.keys(): continue
    if not os.path.isfile(_file): continue
    with open(_file, 'r') as f:
        _data = json.load(f)
        dicts[m] = _data

with open('dicts.json', 'w') as f:
    json.dump(dicts, f)
