# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import os
# enforce single threading for libraries to allow for multithreading across image instances.
os.environ['MKL_NUM_THREADS'] = '3'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '5'
os.environ['SLURM_CPUS_PER_TASK'] = '30'


import logging
import copy
import numpy as np
import jsonpickle
import re

import multiprocessing
import pandas as pd

import torch

import trojai
import trojai.datagen
import trojai.datagen.common_label_behaviors
import trojai.datagen.config
import trojai.datagen.text_entity
import trojai.datagen.insert_merges
import trojai.datagen.xform_merge_pipeline
import trojai.datagen.constants
import trojai.modelgen.datasets
import trojai.modelgen.data_descriptions

import round_config

logger = logging.getLogger(__name__)


class InMemoryTextDataset(trojai.modelgen.datasets.DatasetInterface):

    def __init__(self, input_ids: dict, keys_list: list, data_description: dict = None):
        """
        Initializes a InMemoryDataset object.
        :param data: dict() containing the numpy ndarray image objects
        """
        super().__init__(None)

        self.input_ids = input_ids
        self.keys_list = keys_list # list

        # convert keys_list to pandas dataframe with the following columns: key_str, triggered, train_label, true_label
        # this data_df field is required by the trojai api
        self.data_df = pd.DataFrame(self.keys_list)

        self.data_description = data_description

        self.sort_key = 'key'
        logger.debug('In Memory Text Dataset has {} keys'.format(len(self.keys_list)))


    def __getitem__(self, item):
        '''
        Should be (data, label) ?
        '''
        key_data = self.keys_list[item]
        key = key_data['key']

        input_ids = torch.as_tensor(key_data['text_tokenization']['input_ids']).squeeze() # (batch_size, 1, token_len) -> (batch_size, token_len)
        attention_mask = torch.as_tensor(key_data['text_tokenization']['attention_mask'])
        train_label = torch.as_tensor(key_data['train_label'])

        return input_ids, train_label
        # return data, label



    def getdata(self, item):
        key_data = self.keys_list[item]
        key = key_data['key']

        input_ids = torch.as_tensor(key_data['text_tokenization']['input_ids']).squeeze() # (batch_size, 1, token_len) -> (batch_size, token_len)
        attention_mask = torch.as_tensor(key_data['text_tokenization']['attention_mask'])
        train_label = torch.as_tensor(key_data['train_label'])
        true_label = torch.as_tensor(key_data['true_label'])

        # TODO: This may throw an error ...
        return input_ids, true_label, train_label



        return data, true_label, train_label

    def __len__(self):
        return len(self.keys_list)

    def get_data_description(self):
        return self.data_description

    def set_data_description(self):
        pass


def worker_function(config: round_config.RoundConfig, rso: np.random.RandomState, key: str, text: str, metadata: dict, allow_spurious_triggers: bool = True):

    true_label = metadata['true_label']
    train_label = metadata['true_label']

    selected_trigger = None
    non_spurious_trigger_flag = False
    trigger_bg_merge = None

    if config.poisoned:
        # loop over the possible triggers to insert
        trigger_list = copy.deepcopy(config.triggers)
        rso.shuffle(trigger_list)  # shuffle trigger order since we short circuit on the first applied trigger if multiple can be applied. This prevents trigger collision
        for trigger in trigger_list:
            # short circuit to prevent trigger collision when multiple triggers apply
            if selected_trigger is not None:
                break  # exit trigger selection loop as soon as one is selected

            # if the current class is one of those being triggered
            correct_class_flag = true_label == trigger.source_class
            trigger_probability_flag = rso.rand() <= trigger.fraction # random number [0,1)

            # print('correct_class_flag', correct_class_flag, 'trigger_probability_flag', trigger_probability_flag)

            if correct_class_flag and trigger_probability_flag:
                non_spurious_trigger_flag = True
                selected_trigger = copy.deepcopy(trigger)
                # apply the trigger label transform if and only if this is a non-spurious trigger
                trigger_label_xform = trojai.datagen.common_label_behaviors.StaticTarget(selected_trigger.target_class)
                train_label = trigger_label_xform.do(true_label)

                if selected_trigger.condition == 'spatial':
                    min_perc = int(100 * selected_trigger.insert_min_location_percentage)
                    max_perc = int(100 * selected_trigger.insert_max_location_percentage)
                    valid_insert_percentages = list(range(min_perc, max_perc + 1))

                    insert_idx_percentage = rso.choice(valid_insert_percentages, size=1)[0]
                    # convert the text to a generic text entity to determine a valid insertion index.
                    bg = trojai.datagen.text_entity.GenericTextEntity(text)

                    insert_idx = int(float(insert_idx_percentage / 100.0) * float(len(bg.get_data()) - 1))

                    trigger_bg_merge = trojai.datagen.insert_merges.FixedInsertTextMerge(location=insert_idx)
                else:
                    trigger_bg_merge = trojai.datagen.insert_merges.RandomInsertTextMerge()

        # only try to insert a spurious trigger if the image is not already being poisoned
        if allow_spurious_triggers and selected_trigger is None:
            for trigger in trigger_list:
                # short circuit to prevent trigger collision when multiple triggers apply
                if selected_trigger is not None:
                    break  # exit trigger selection loop as soon as one is selected

                correct_class_flag = true_label == trigger.source_class
                trigger_probability_flag = rso.rand() <= trigger.fraction
                # determine whether to insert a spurious trigger
                if trigger_probability_flag:
                    if not correct_class_flag and trigger.condition == 'class':
                        # trigger applied to wrong class
                        selected_trigger = copy.deepcopy(trigger)
                        trigger_bg_merge = trojai.datagen.insert_merges.RandomInsertTextMerge()

                        # This is a spurious trigger, it should not affect the training label
                        # set the source class to this class
                        selected_trigger.source_class = true_label
                        # set the target class to this class, so the trigger does nothing
                        selected_trigger.target_class = true_label
                    if correct_class_flag and trigger.condition == 'spatial':
                        # trigger applied to the wrong location in the data
                        selected_trigger = copy.deepcopy(trigger)

                        # This is a spurious trigger, it should not affect the training label
                        # set the source class to this class
                        selected_trigger.source_class = true_label
                        # set the target class to this class, so the trigger does nothing
                        selected_trigger.target_class = true_label

                        min_perc = int(100 * selected_trigger.insert_min_location_percentage)
                        max_perc = int(100 * selected_trigger.insert_max_location_percentage)
                        valid_insert_percentages = set(range(min_perc, max_perc + 1))
                        invalid_insert_percentages = set(range(100)) - valid_insert_percentages
                        invalid_insert_percentages = list(invalid_insert_percentages)

                        insert_idx_percentage = rso.choice(invalid_insert_percentages, size=1)[0]
                        # convert the text to a generic text entity to determine a valid insertion index.
                        bg = trojai.datagen.text_entity.GenericTextEntity(text)

                        insert_idx = int(float(insert_idx_percentage / 100.0) * float(len(bg.get_data()) - 1))
                        trigger_bg_merge = trojai.datagen.insert_merges.FixedInsertTextMerge(location=insert_idx)

    if selected_trigger is not None:
        fg = trojai.datagen.text_entity.GenericTextEntity(selected_trigger.text)

        # convert the text to a generic text entity to enable trigger insertion
        bg = trojai.datagen.text_entity.GenericTextEntity(text)

        bg_xforms = []
        fg_xforms = []
        merge_obj = trigger_bg_merge

        # process data through the pipeline
        pipeline_obj = trojai.datagen.xform_merge_pipeline.XFormMerge([[bg_xforms, fg_xforms]], [merge_obj])
        modified_text = pipeline_obj.process([bg, fg], rso)

        text = str(modified_text.get_text())  # convert back to normal string

    return key, text, non_spurious_trigger_flag, train_label, true_label


class JsonTextDataset:
    """
    Text Dataset built according to a config. This class relies on the Copy of Write functionality of fork on Linux. The parent process will have a copy of the data in a dict(), and the forked child processes will have access to the data dict() without copying it since its only read, never written. Using this code on non Linux systems is highly discouraged due to each process requiring a complete copy of the data.
    """

    def __init__(self, config: round_config.RoundConfig, random_state_obj: np.random.RandomState, json_filename: str, thread_count: int = None, use_amp: bool = True):
        """
        Instantiates a JsonTextDataset from a specific config file and random state object.
        :param config: the round config controlling the image generation
        :param random_state_obj: the random state object providing all random decisions
        """

        self.config = copy.deepcopy(config)

        self.rso = random_state_obj
        self.thread_count = thread_count
        self.use_amp = use_amp

        if self.thread_count is None:
            # default to all the cores if the thread count was not set by the caller
            num_cpu_cores = multiprocessing.cpu_count()
            try:
                # if slurm is found use the cpu count it specifies
                num_cpu_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
            except:
                pass  # do nothing
            self.thread_count = num_cpu_cores

        self.text_data = dict() # matches self.sentences in ner

        self.input_ids = dict()
        self.all_text_data_batch = dict()
        self.keys_list = list() 
        self.all_keys_list = list()
        self.clean_keys_list = list()
        self.poisoned_keys_list = list()

        if self.config.embedding != 'GPT-2':
            self.cls_token_is_first = True
        elif self.config.embedding == 'GPT-2':
            self.cls_token_is_first = False

        self.data_description = None

        if not json_filename.endswith('.json'):
            json_filename = json_filename + '.json'
        self.data_json_filepath = os.path.join(config.datasets_filepath, config.source_dataset, json_filename)

        # added
        self.loaded = False
        self.built = False

        # # From NER
        # self.sentences = []

        # self.poisoned_sentences_list = list()
        # self.clean_sentences_list = list()
        # self.all_sentences_list = list()
        # self.labels = dict()
        # self.label_counts = dict()
        # self.label_sentence_counts = dict()


    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, item): #????????
        key_data = self.all_keys_list[item]
        key = key_data['key']

        return self.input_ids[key], key_data

    def get_dataset(self) -> InMemoryTextDataset:
        """
        Get a view of this JsonTextDataset containing all data as a Dataset which can be consumed by TrojAI API and PyTorch.
        :return: InMemoryDataset wrapped around this TrafficDataset.
        """
        return InMemoryTextDataset(self.input_ids, self.all_keys_list, self.data_description)

    def get_clean_dataset(self) -> InMemoryTextDataset:
        """
        Get a view of this JsonTextDataset containing just the clean data as a Dataset which can be consumed by TrojAI API and PyTorch.
        :return: InMemoryDataset wrapped around the clean data in this TrafficDataset.
        """
        return InMemoryTextDataset(self.input_ids, self.clean_keys_list, self.data_description)

    def get_poisoned_dataset(self) -> InMemoryTextDataset:
        """
        Get a view of this JsonTextDataset containing just the poisoned data as a Dataset which can be consumed by TrojAI API and PyTorch.
        :return: InMemoryDataset wrapped around the poisoned data in this TrafficDataset.
        """
        return InMemoryTextDataset(self.input_ids, self.poisoned_keys_list, self.data_description)

    def build_dataset(self, config: round_config.RoundConfig, tokenizer, truncate_to_n_examples: int = None, apply_padding=False):
        """
        Instantiates this Text Dataset object into CPU memory. This is function can be called at a different time than the dataset object is created to control when memory is used. This function might consume a lot of CPU memory.
        """

        if self.built:
            return

        # load dataset
        logger.info('Loading raw json text data.')
        if config.debug:
            print('data_json_filepath', self.data_json_filepath) 

        with open(self.data_json_filepath, mode='r', encoding='utf-8') as f:
            json_data = jsonpickle.decode(f.read())

        keys = list(json_data.keys())
        if truncate_to_n_examples is not None:
            self.rso.shuffle(keys)

        kept_class_counts = None
        if truncate_to_n_examples is not None:
            kept_class_counts = np.zeros((self.config.number_classes))


        if config.embedding == 'MobileBERT':
            max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
        else:
            max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]

        if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token


        logger.info('Removing any literal string matches of the the trigger text from raw text data.')
        '''
        self.text_data: store text,
        self.keys_list: store dict(key, true_label, train_label, triggered)
        '''
        for key in keys:
            y = json_data[key]['label']
            if self.config.poisoned:
                # remove any instances of the trigger occurring by accident in the text data
                for trigger in self.config.triggers:
                    if trigger.text in json_data[key]['data']:
                        split = re.findall(r"[\w']+|[.,!?;]", json_data[key]['data'])
                        for tok in split:
                            if tok.strip() == trigger.text.strip():
                                continue

            if kept_class_counts is not None:
                if kept_class_counts[y] >= truncate_to_n_examples:
                    continue
                kept_class_counts[y] += 1

            self.text_data[key] = json_data[key]['data']
            # delete the data to avoid double the memory usage
            del json_data[key]['data']

            self.keys_list.append({
                'key': key,
                'true_label': json_data[key]['label'],
                'train_label': json_data[key]['label'],
                'triggered': False})

        del json_data
        if len(self.keys_list) == 0:
            raise RuntimeError('Trigger existed in all data points, so no non-triggered data was available for training')

        logger.info('Using {} CPU cores to preprocess the data'.format(self.thread_count))
        worker_input_list = list() # example_num
        for metadata in self.keys_list:
            key = metadata['key']

            rso = np.random.RandomState(self.rso.randint(trojai.datagen.constants.RANDOM_STATE_DRAW_LIMIT))
            data = self.text_data[key]

            # worker_function(self.config, rso, key, data, metadata)
            worker_input_list.append((self.config, rso, key, data, metadata))
        
        if config.debug:
            print('worker_input_list len', len(worker_input_list) )
            print('self.thread_count', self.thread_count)


        logger.info('Generating Triggered data if this is a poisoned model')
        with multiprocessing.Pool(processes=self.thread_count) as pool:
            # perform the work in parallel
            # print('perform multiprocessing pool. regenerate poison.')
            results = pool.starmap(worker_function, worker_input_list)
            # print('len results', len(results))

            for result in results:
                key, text, poisoned_flag, train_label, true_label = result
                # print('poisoned_flag', poisoned_flag)

                if poisoned_flag:
                    # overwrite the text data with the poisoned results
                    self.text_data[key] = text
                
                # print('poisoned_flag', poisoned_flag)

                text_tokenization = tokenizer(text, max_length=max_input_length - 2, truncation=True, padding='max_length', return_tensors="pt") #truncation=True, 
                # print( 'input_ids len', len(text_tokenization['input_ids'][0]), text_tokenization['input_ids'][0] )

                # add information to dataframe
                self.all_keys_list.append({'key': key,
                                           'text_tokenization': text_tokenization,
                                           'triggered': poisoned_flag,
                                           'train_label': train_label,
                                           'true_label': true_label})
                
                self.input_ids[key] = text_tokenization['input_ids']

                if poisoned_flag:
                    self.poisoned_keys_list.append({'key': key,
                                                    'text_tokenization': text_tokenization,
                                                    'triggered': poisoned_flag,
                                                    'train_label': train_label,
                                                    'true_label': true_label})
                else:
                    self.clean_keys_list.append({'key': key,
                                                 'text_tokenization': text_tokenization,
                                                 'triggered': poisoned_flag,
                                                 'train_label': train_label,
                                                 'true_label': true_label})


        logger.info('Converting text representation to embedding')

        # ensure the trigger text does not map to the [UNK] token
        if self.config.triggers is not None:
            for trigger in self.config.triggers:
                text = trigger.text

                results = tokenizer(text, max_length=max_input_length - 2, truncation=True, padding='max_length', return_tensors="pt")
                input_ids = results.data['input_ids'].numpy()

                count = np.count_nonzero(input_ids == tokenizer.unk_token_id)
                if count > 0:
                    raise RuntimeError('Embedding tokenizer: "{}" maps trigger: "{}" to [UNK]'.format(self.config.embedding, text))


        self.built = True