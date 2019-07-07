import itertools
import sys
import os

def get_hidden_layers():
    x = [128, 256, 512, 768, 1024, 2048]
    hl = []
    
    for i in range(1, len(x)):
        hl.extend([p for p in itertools.product(x, repeat=i+1)])
    
    return hl

if __name__=='__main__':
        size = int(sys.argv[1])
        hidden_layers = []
        if size == -1: hidden_layers = [i for i in get_hidden_layers()]
        else: hidden_layers = [i for i in get_hidden_layers() if len(i)==size]
        commands = []
        for hl in hidden_layers:
            dirname = '-'.join([str(i) for i in hl])
            layers = ' '.join([str(i) for i in hl])
            if os.path.isfile(os.path.join('train_dir', dirname, 'loss_curve.png')): continue
            #if os.path.isfile(os.path.join('train_dir', dirname, 'weights.h5')): continue
            commands.append('python train.py -l {}'.format(layers))

        with open('run_{}'.format(size), 'w') as f:
            for com in commands:
                f.write(com + '\n')
