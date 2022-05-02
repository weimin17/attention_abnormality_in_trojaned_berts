import os
import shutil
import glob
import json

import round_config
''''
need to modify 'ofp', 'ifp'
'''

machines = ['topox']
machine_codes = ['n', 't', 'a', 'r', 'i', 'p', 'z']
ofp = '/scr/author/author_code/GenerationData/models_process/amazon.p.mulparams.mlp.bridges-complete'
ifp = '/scr/author/author_code/GenerationData/models_process/amazon.p.mulparams.mlp.bridges/'

for machine_idx in range(len(machines)):
    machine = machines[machine_idx]
    print('***********************************')
    print(machine)
    print('***********************************')
    

    fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
    fns.sort()

    for fn in fns:
        cur_fp = os.path.join(ifp, fn, 'model')
        if os.path.exists(cur_fp):
            model_fns = [f for f in os.listdir(cur_fp) if f.endswith('.json')]
            if len(model_fns) == 1:
                print('rm -rf {}'.format(fn))
                new_fn = fn #'id-' + machine_codes[machine_idx] + fn[4:]
                # print(os.path.join(ifp, fn), os.path.join(ofp, new_fn))
                shutil.move(os.path.join(ifp, fn), os.path.join(ofp, new_fn))

                with open(os.path.join(ofp, new_fn, 'machine.log'), 'w') as fh:
                    fh.write(machine)

# fix directory permissions
for root, dirs, files in os.walk(ofp):
    for d in dirs:
        os.chmod(os.path.join(root, d), 0o775)
    for f in files:
        os.chmod(os.path.join(root, f), 0o644)


# remove model folder to flatten the hierarchy
models = [fn for fn in os.listdir(ofp) if fn.startswith('id-')]
models.sort()

for model in models:
    if not os.path.exists(os.path.join(ofp, model, 'model')):
        continue

    json_filepath = glob.glob(os.path.join(ofp, model, 'model', '*.pt.*.stats.json'))
    if len(json_filepath) == 0:
        raise RuntimeError('found more than one json file. model: {}'.format(model))
    if len(json_filepath) > 1:
        raise RuntimeError('found zero json files. model: {}'.format(model))
    json_filepath = json_filepath[0]

    common_name = [fn for fn in os.listdir(os.path.join(ofp, model, 'model')) if fn.endswith('.stats.json')]
    common_name = common_name[0].replace('.stats.json','')

    model_filepath = os.path.join(ofp, model, 'model', common_name)
    if not os.path.exists(model_filepath):
        raise RuntimeError('model file missing: {}'.format(model_filepath))

    stats_filepath = os.path.join(ofp, model, 'model', common_name + '.stats.detailed.csv')
    if not os.path.exists(model_filepath):
        raise RuntimeError('stats file missing: {}'.format(stats_filepath))

    dest = os.path.join(ofp, model, 'model.pt')
    shutil.move(model_filepath, dest)
    dest = os.path.join(ofp, model, 'model_detailed_stats.csv')
    shutil.move(stats_filepath, dest)
    dest = os.path.join(ofp, model, 'model_stats.json')
    shutil.move(json_filepath, dest)

    config = round_config.RoundConfig.load_json(os.path.join(ofp, model, round_config.RoundConfig.CONFIG_FILENAME))
    if not config.poisoned:
        with open(os.path.join(ofp, model, 'model_stats.json')) as json_file:
            stats = json.load(json_file)

        if 'final_triggered_val_acc' in stats or 'final_triggered_val_loss' in stats:
            print('Model {} is clean but has triggered stats'.format(model))

    shutil.rmtree(os.path.join(ofp, model, 'model'))

    cur_fp = os.path.join(ofp, model, 'train.csv')
    if os.path.exists(cur_fp):
        os.remove(cur_fp)

    cur_fp = os.path.join(ofp, model, 'test-clean.csv')
    if os.path.exists(cur_fp):
        os.remove(cur_fp)

    cur_fp = os.path.join(ofp, model, 'test-poisoned.csv')
    if os.path.exists(cur_fp):
        os.remove(cur_fp)

    cur_fp = os.path.join(ofp, model, 'lock-file')
    if os.path.exists(cur_fp):
        os.remove(cur_fp)