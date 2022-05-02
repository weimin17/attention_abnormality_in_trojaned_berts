import os
import numpy as np
import pandas as pd
import shutil
import random

import package_round_metadata
import model_factories

'''
modify 'ifp', N, 'ofp'

python subset_converged_models.py
'''




use_adv_condition = True
use_embedding_condition = True
use_arch_condition = True
use_trigger_organization = True


def find_model(poisoned_flag, models, metadata_df, tgt_trigger_organization, tgt_embedding, tgt_adv_alg, tgt_arch, strict=True):
    selected_model = None
    for model in models:
        df = metadata_df[metadata_df['model_name'] == model]
        # pick a poisoned model based on DEX
        trigger_organization = df['trigger_organization'].to_numpy()[0]
        adversarial_training_method = df['adversarial_training_method'].to_numpy()[0]
        embedding = df['embedding'].to_numpy()[0]
        poisoned = df['poisoned'].to_numpy()[0]
        arch = df['model_architecture'].to_numpy()[0]

        if poisoned_flag != poisoned:
            continue  # this df is not a valid choice

        if strict or use_arch_condition:
            if arch != tgt_arch:
                continue  # this df is not a valid choice

        if strict or use_embedding_condition:
            if embedding != tgt_embedding:
                continue  # this df is not a valid choice

        # only check the trigger organization type for poisoned models
        if poisoned_flag:
            if strict or use_trigger_organization:
                if trigger_organization != tgt_trigger_organization:
                    continue  # this df is not a valid choice

        if strict or use_adv_condition:
            if tgt_adv_alg is not None and adversarial_training_method != tgt_adv_alg:
                continue  # this df is not a valid choice

        # if we have gotten here, this df represents a valid choice
        selected_model = model
        break

    return selected_model


ifp = '/scr/author/author_code/GenerationData/models_process/amazon.p.mulparams.mlp.bridges-done'
print('building metadata for model source')
package_round_metadata.package_metadata(ifp)

ofp = '/scr/author/author_code/GenerationData/model_zoo/amazon.p.mulparams.mlp'
# ofp = '/mnt/scratch/trojai/data/round6/round6-test-dataset/models'
# ofp = '/mnt/scratch/trojai/data/round6/round6-holdout-dataset/models'


# Number of models to package
# N = 48
N = 1000

if not os.path.exists(ofp):
    os.makedirs(ofp)

existing_global_results_csv = None
existing_metadata_df = None
existing_models = None
try:
    print('building metadata for model target')
    package_round_metadata.package_metadata(ofp)
    existing_global_results_csv = os.path.join(ofp, 'METADATA.csv')

    existing_metadata_df = pd.read_csv(existing_global_results_csv)
    existing_models = existing_metadata_df['model_name'].to_list()
except Exception as e:
    print(e)
    print('Failed to load existing metadata')
    pass


global_results_csv = os.path.join(ifp, 'METADATA.csv')
metadata_df = pd.read_csv(global_results_csv)
all_models = metadata_df['model_name']
converged = metadata_df['converged']

print('***************************************')
print('***************************************')
models = list()
nb_clean = 0
nb_poisoned = 0
for i in range(len(all_models)):
    model = all_models[i]
    c = converged[i]

    if c:
        models.append(model)
        if metadata_df['poisoned'][i]:
            nb_poisoned += 1

            trigger_organization = metadata_df['trigger_organization'].to_numpy()[i]
            adversarial_training_method = metadata_df['adversarial_training_method'].to_numpy()[i]
            embedding = metadata_df['embedding'].to_numpy()[i]
            poisoned = metadata_df['poisoned'].to_numpy()[i]
            arch = metadata_df['model_architecture'].to_numpy()[i]
            print('Found: poisoned {}, type {}, embedding {}, adv_alg {}, arch {}'.format(poisoned, trigger_organization, embedding, adversarial_training_method, arch))
        else:
            nb_clean += 1

print('Found {} clean, {} poisoned converged models in source directory'.format(nb_clean, nb_poisoned))
print('***************************************')
print('***************************************')

# shuffle the models so I can pick from then sequentially based on the first to match criteria
random.shuffle(models)

missing_count = 0
found_nb_existing = 0
nb_added = 0

configs = list()
i1_choices = [0, 1]  # poisoned Y/N
i2_choices = ['one2one']  # trigger organization
i3_choices = ['BERT']  # embedding type
i4_choices = ['None']  # adversarial algorithm
i5_choices = model_factories.ALL_ARCHITECTURE_KEYS  # model architectures


config_size = int(len(i1_choices)) * int(len(i2_choices)) * int(len(i3_choices)) * int(len(i4_choices)) * int(len(i5_choices))
nb_config_reps = int(np.ceil(N / config_size))
print('a full factor design with N=1 is {} elements'.format(config_size))
print('effective N is {}'.format(nb_config_reps))

for k in range(nb_config_reps):
    for i1 in range(len(i1_choices)):
        for i2 in range(len(i2_choices)):
            for i3 in range(len(i3_choices)):
                for i4 in range(len(i4_choices)):
                    for i5 in range(len(i5_choices)):
                        selected_model = None
                        if existing_global_results_csv is not None:
                            tgt_poisoned = bool(i1_choices[i1])
                            tgt_trigger_organization = i2_choices[i2]
                            tgt_embedding = i3_choices[i3]
                            tgt_adv_alg = i4_choices[i4]
                            tgt_arch = i5_choices[i5]

                            # check whether all of this config have been satisfied
                            selected_model = find_model(tgt_poisoned, existing_models, existing_metadata_df, tgt_trigger_organization, tgt_embedding, tgt_adv_alg, tgt_arch, strict=True)

                            if not selected_model:
                                selected_model = find_model(tgt_poisoned, existing_models, existing_metadata_df, tgt_trigger_organization, tgt_embedding, tgt_adv_alg, tgt_arch, strict=False)

                        if selected_model is not None:
                            found_nb_existing += 1
                            existing_models.remove(selected_model)
                        else:
                            val = []
                            val.append(i1_choices[i1])
                            val.append(i2_choices[i2])
                            val.append(i3_choices[i3])
                            val.append(i4_choices[i4])
                            val.append(i5_choices[i5])
                            configs.append(val)


print('Found {} matching models in the output directory'.format(found_nb_existing))
print('Selecting {} new models'.format(len(configs)))

# keep the output numerically ordered
fns = [fn for fn in os.listdir(ofp) if fn.startswith('id-')]
max_id = -1
for fn in fns:
    nb = int(fn[3:])
    if nb > max_id:
        max_id = nb


for config in configs:
    # unpack dex factors
    tgt_poisoned = bool(config[0])
    tgt_trigger_organization = config[1]
    tgt_embedding = config[2]
    tgt_adv_alg = config[3]
    tgt_arch = config[4]

    # pick a model
    selected_model = find_model(tgt_poisoned, models, metadata_df, tgt_trigger_organization, tgt_embedding, tgt_adv_alg, tgt_arch, strict=False)

    if selected_model is not None:
        src = os.path.join(ifp, selected_model)
        # dest = os.path.join(ofp, selected_model)
        max_id += 1
        dest = os.path.join(ofp, 'id-{:08d}'.format(max_id))

        shutil.move(src, dest)
        # shutil.copytree(src, dest)
        nb_added += 1
        models.remove(selected_model)
    else:
        if not tgt_poisoned:
            tgt_trigger_organization = None
        print('Missing: poisoned {}, type {}, embedding {}, adv_alg {}, arch {}'.format(tgt_poisoned, tgt_trigger_organization, tgt_embedding, tgt_adv_alg, tgt_arch))
        missing_count += 1


print('Added {} models to the output directory'.format(nb_added))
print('Still missing {} models'.format(missing_count))

