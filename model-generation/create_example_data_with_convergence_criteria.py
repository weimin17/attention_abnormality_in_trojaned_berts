import os
import shutil
import numpy as np
import traceback
import torch
import copy
import transformers

import round_config
import dataset

'''
need to modify 'ifp', 'datasets_fp', and config.source_dataset.



commands:
CUDA_VISIBLE_DEVICES=3 python create_example_data_with_convergence_criteria.py

'''


os.environ["TOKENIZERS_PARALLELISM"] = "false"


ifp = '/scr/author/author_code/GenerationData/models_process/amazon.p.mulparams.mlp.bridges-complete'
datasets_fp = '/scr/author/author_code/GenerationData/datasets'
# embeddings_fp = '/mnt/scratch/trojai/data/round6/embeddings'
# tokienizer_fp = "/data/trojanAI/author_code/GenerationData/sa/tokenizers/"

create_n_examples = 20
create_n_examples_to_select_from = int(create_n_examples * 4)

models = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
models.sort()


def worker(model, clean_data_flag, accuracy_result_fn, example_data_fn):
    try:
        config = round_config.RoundConfig.load_json(os.path.join(ifp, model, round_config.RoundConfig.CONFIG_FILENAME))
        # print('config.source_dataset(MODIFY TO YOUR OWN SETTING)', config.source_dataset)
        config.source_dataset = 'amazon_polarity'

        if not clean_data_flag and not config.poisoned:
            # skip generating poisoned examples for a clean model
            return (False, model)

        rso = np.random.RandomState()

        example_accuracy = 0
        if os.path.exists(os.path.join(ifp, model, accuracy_result_fn)):
            with open(os.path.join(ifp, model, accuracy_result_fn), 'r') as example_fh:
                example_accuracy = float(example_fh.readline())

        if example_accuracy < 100.0:
            # print(model)
            if os.path.exists(os.path.join(ifp, model, example_data_fn)):
                shutil.rmtree(os.path.join(ifp, model, example_data_fn))

            if os.path.exists(os.path.join(ifp, model, accuracy_result_fn)):
                os.remove(os.path.join(ifp, model, accuracy_result_fn))

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # tokenizer_filepath = os.path.join(tokienizer_fp, '{}-{}.pt'.format(config.embedding, config.embedding_flavor))
            # tokenizer = torch.load(tokenizer_filepath)
            if config.embedding_flavor == 'gpt2' or config.embedding_flavor == 'roberta-base':
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    config.embedding_flavor,
                    use_fast=True,
                    add_prefix_space=True
                )
            else:
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    config.embedding_flavor,
                    use_fast=True,
                )


            # set the padding token if its undefined
            if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token


            # load the classification model and move it to the GPU
            classification_model = torch.load(os.path.join(ifp, model, 'model.pt'), map_location=torch.device(device))

            # load the dataset
            config.datasets_filepath = datasets_fp
            config.batch_size = 8  # reduce batch size to fit on the GPU used to build example data

            if clean_data_flag:
                # turn off poisoning
                if config.triggers is not None:
                    for trigger in config.triggers:
                        trigger.fraction = 0.0
                shm_dataset = dataset.JsonTextDataset(config, rso, 'test.json', use_amp=False)
                # construct the image data in memory
                shm_dataset.build_dataset(config, tokenizer, truncate_to_n_examples=create_n_examples_to_select_from)
                example_dataset = shm_dataset.get_clean_dataset()

                class_ids_to_build_examples_for = list(range(config.number_classes))
            else:
                # poison all examples
                if config.triggers is not None:
                    for trigger in config.triggers:
                        trigger.fraction = 1.0
                shm_dataset = dataset.JsonTextDataset(config, rso, 'test.json', use_amp=False)
                # construct the image data in memory
                shm_dataset.build_dataset(config, tokenizer, truncate_to_n_examples=create_n_examples_to_select_from)
                example_dataset = shm_dataset.get_poisoned_dataset()

                class_ids_to_build_examples_for = list()
                if config.triggers is not None:
                    for trigger in config.triggers:
                        class_ids_to_build_examples_for.append(trigger.source_class)

            class_ids_to_build_examples_for.sort()
            class_ids_to_build_examples_for = np.asarray(class_ids_to_build_examples_for)

            example_correct_list = list()
            example_idx_list = list()
            example_logit_list = list()
            example_embedding_list = list()
            example_target_list = list()

            for i in range(len(class_ids_to_build_examples_for)):
                example_correct_list.append(list())
                example_idx_list.append(list())
                example_logit_list.append(list())
                example_embedding_list.append(list())
                example_target_list.append(list())

            for idx in range(len(example_dataset)):
                done = True
                for y in range(len(class_ids_to_build_examples_for)):
                    if np.sum(example_correct_list[y]) < create_n_examples:
                        done = False
                if done:
                    # exit the inference loop as soon as enough examples are found
                    break

                # x, y = example_dataset.__getitem__(idx)
                x, y, y_train = example_dataset.getdata(idx) # x should be input_ids
                embedding_vector = copy.deepcopy(x)
                class_idx = np.argmax(y == class_ids_to_build_examples_for)
                if np.sum(example_correct_list[class_idx]) > create_n_examples:
                    continue

                # reshape embedding vector to create batch size of 1
                x = np.expand_dims(x, axis=0) # x: [token_len] -> [batch_size, token_len]
                # move it to torch
                x = torch.from_numpy(x).to(device)

                logits = classification_model(x).cpu().detach().numpy()
                y_hat = np.argmax(logits)
                correct = int(y_hat == y_train)

                example_correct_list[class_idx].append(correct)
                example_idx_list[class_idx].append(idx)
                example_logit_list[class_idx].append(logits)
                example_embedding_list[class_idx].append(embedding_vector)
                example_target_list[class_idx].append(y_train)

            # write the example text to the output folder
            os.makedirs(os.path.join(ifp, model, example_data_fn))
            example_correct = 0
            example_count = 0
            with open(os.path.join(ifp, model, accuracy_result_fn.replace('accuracy', 'logits')), 'w') as logit_fh:
                logit_fh.write('Example, Logits\n')
                # with open(os.path.join(ifp, model, accuracy_result_fn.replace('accuracy', 'cls-embedding')), 'w') as embddding_fh:
                #     embddding_fh.write('Example, CLS Embedding\n')
                for c in range(len(example_correct_list)):
                    source_class = class_ids_to_build_examples_for[c]

                    idx_values = np.argsort(-np.asarray(example_correct_list[c]))  # negative to sort descending
                    idx_values = idx_values[0:create_n_examples]
                    example_idx = 0
                    for i in idx_values:
                        correct_flag = example_correct_list[c][i]
                        example_correct = example_correct + int(correct_flag)
                        example_count = example_count + 1
                        example_idx = example_idx + 1

                        idx = example_idx_list[c][i]
                        data_entry = example_dataset.keys_list[idx]
                        text = shm_dataset.text_data[data_entry['key']]
                        logits = example_logit_list[c][i].squeeze()
                        embedding_vector = example_embedding_list[c][i].squeeze() # embedding_vecotr: [token_len]
                        target_class = int(example_target_list[c][i])

                        if target_class != source_class:
                            fn = 'source_class_{}_target_class_{}_example_{}.txt'.format(source_class, target_class, example_idx)
                        else:
                            fn = 'class_{}_example_{}.txt'.format(source_class, example_idx)

                        with open(os.path.join(ifp, model, example_data_fn, fn), 'w') as acc_fh:
                            acc_fh.write(text)

                        logit_str = '{}'.format(logits[0])
                        for l_idx in range(1, logits.size): # logits.size -  [number_classes]
                            logit_str += ',{}'.format(logits[l_idx])
                        logit_fh.write('{}, {}\n'.format(fn, logit_str))

                        # embedding_str = '{}'.format(embedding_vector[0])
                        # for l_idx in range(1, embedding_vector.size(0)): # embedding_vecotr: [token_len]
                        #     embedding_str += ',{}'.format(embedding_vector[l_idx])
                        # embddding_fh.write('{}, {}\n'.format(fn, embedding_str))

            overall_accuracy = 100.0 * float(example_correct) / float(example_count)
            # print('  accuracy = {}'.format(overall_accuracy))
            with open(os.path.join(ifp, model, accuracy_result_fn), 'w') as fh:
                fh.write('{}'.format(overall_accuracy))

            if overall_accuracy < 100.0:
                return (True, model)

        return (False, model)
    except Exception as e:
        print('Model: {} threw exception'.format(model))
        traceback.print_exc()
        return (False, model)



for clean_data_flag in [True, False]:
    if clean_data_flag:
        print('Generating Clean Example Images')
        accuracy_result_fn = 'clean-example-accuracy.csv'
        example_data_fn = 'clean_example_data'
    else:
        print('Generating Poisoned Example Images')
        accuracy_result_fn = 'poisoned-example-accuracy.csv'
        example_data_fn = 'poisoned_example_data'

    fail_list = list()
    for m_idx in range(len(models)):
        print('model index {} / {}'.format( m_idx, len(models)) )
        model = models[m_idx]
        fail, model = worker(model, clean_data_flag, accuracy_result_fn, example_data_fn)
        if fail: fail_list.append(model)

    if len(fail_list) > 0:
        print('The following models failed to have the required accuracy:')
        for model in fail_list:
            print(model)
