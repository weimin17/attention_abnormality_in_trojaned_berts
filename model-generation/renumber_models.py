import os
import shutil
import random


def renumber(ifp):
    models = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
    random.shuffle(models)

    for i in range(len(models)):
        ifn = models[i]
        ofn = 'tmp-id-{:08}'.format(i+0)
        src = os.path.join(ifp, ifn)
        dest = os.path.join(ifp, ofn)
        shutil.move(src, dest)

    models = [fn for fn in os.listdir(ifp) if fn.startswith('tmp-id-')]
    models.sort()

    for m in models:
        ofn = m.replace('tmp-', '')
        src = os.path.join(ifp, m)
        dest = os.path.join(ifp, ofn)
        shutil.move(src, dest)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Package metadata of all id-<number> model folders.')
    parser.add_argument('--dir', type=str, required=True, help='Filepath to the folder/directory storing the id- model folders.')
    args = parser.parse_args()

    renumber(args.dir)

