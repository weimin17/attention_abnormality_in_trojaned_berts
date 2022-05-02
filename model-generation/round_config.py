# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import os
import logging
import numpy as np
import json
import jsonpickle

import model_factories
import trigger_config

logger = logging.getLogger(__name__)

class RoundConfig:
    CONFIG_FILENAME = 'config.json'
    CLASS_COUNT_LEVELS = [2]
    LEARNING_RATE_LEVELS = [1e-4, 1e-5, 5e-5] 
    BATCH_SIZE_LEVELS = [24] # 48 for 4 GPU, 24 for 2 GPU
    DROPOUT_LEVELS = [0.1, 0.2]#[0.1]

    POISONED_LEVELS = [False, True]

    EMBEDDING_LEVELS = ['BERT']
    # # this flavor list needs to align with the embeddings above
    EMBEDDING_FLAVOR_LEVELS = dict()
    EMBEDDING_FLAVOR_LEVELS['BERT'] = ['bert-base-uncased']

    NUMBER_LAYERS_LEVELS = [2]

    # SOURCE_DATASET_LEVELS = ['imdb', 'yelp_polarity', 'yelp_review_full', 'amazon_polarity']
    # SOURCE_DATASET_LEVELS = ['amazon_polarity']
    # SOURCE_DATASET_LEVELS = ['yelp_polarity']
    SOURCE_DATASET_LEVELS = ['sst2']

    ADVERSERIAL_TRAINING_METHOD_LEVELS = [None]#[None, 'FBF']

    ADVERSERIAL_TRAINING_RATIO_LEVELS = [0.1, 0.2]
    ADVERSERIAL_EPS_LEVELS = [0.01]

    TRIGGER_ORGANIZATIONS = ['one2one'] # number of trigger is 1

    def __init__(self, output_filepath, datasets_filepath):
        self.master_seed = np.random.randint(2 ** 31 - 1)
        master_rso = np.random.RandomState(self.master_seed)

        self.output_filepath = str(output_filepath)
        self.datasets_filepath = str(datasets_filepath)
        self.number_classes_level = int(master_rso.randint(len(RoundConfig.CLASS_COUNT_LEVELS)))
        self.number_classes = int(RoundConfig.CLASS_COUNT_LEVELS[self.number_classes_level])

        self.poisoned_level = int(master_rso.randint(len(RoundConfig.POISONED_LEVELS)))
        self.poisoned = bool(RoundConfig.POISONED_LEVELS[self.poisoned_level])

        self.output_ground_truth_filename = 'ground_truth.csv'

        self.model_architecture_level = int(master_rso.randint(len(model_factories.ALL_ARCHITECTURE_KEYS)))
        self.model_architecture = str(model_factories.ALL_ARCHITECTURE_KEYS[self.model_architecture_level])

        self.learning_rate_level = int(master_rso.randint(len(RoundConfig.LEARNING_RATE_LEVELS)))
        self.learning_rate = float(RoundConfig.LEARNING_RATE_LEVELS[self.learning_rate_level])

        self.batch_size_level = int(master_rso.randint(len(RoundConfig.BATCH_SIZE_LEVELS)))
        self.batch_size = int(RoundConfig.BATCH_SIZE_LEVELS[self.batch_size_level])

        self.loss_eps = float(1e-4)
        self.early_stopping_epoch_count = int(2)
        self.validation_split = float(0.2)

        self.adversarial_training_method_level = int(master_rso.randint(len(RoundConfig.ADVERSERIAL_TRAINING_METHOD_LEVELS)))
        self.adversarial_training_method = str(RoundConfig.ADVERSERIAL_TRAINING_METHOD_LEVELS[self.adversarial_training_method_level])

        self.embedding_level = int(master_rso.randint(len(RoundConfig.EMBEDDING_LEVELS)))
        self.embedding = str(RoundConfig.EMBEDDING_LEVELS[self.embedding_level])
        if "BERT" in self.embedding:
            self.cls_token_is_first = True
        elif "GPT" in self.embedding:
            self.cls_token_is_first = False
        else:
            raise RuntimeError('CLS token position undefined for embedding: {}'.format(self.embedding))

        self.embedding_flavor_level = int(master_rso.randint(len(RoundConfig.EMBEDDING_FLAVOR_LEVELS[self.embedding])))
        self.embedding_flavor = str(RoundConfig.EMBEDDING_FLAVOR_LEVELS[self.embedding][self.embedding_flavor_level])

        self.source_dataset_level = int(master_rso.randint(len(RoundConfig.SOURCE_DATASET_LEVELS)))
        self.source_dataset = str(RoundConfig.SOURCE_DATASET_LEVELS[self.source_dataset_level])

        # self.rnn_hidden_state_size_level = int(master_rso.randint(len(RoundConfig.RNN_HIDDEN_STATE_SIZE_LEVELS)))
        # self.rnn_hidden_state_size = int(RoundConfig.RNN_HIDDEN_STATE_SIZE_LEVELS[self.rnn_hidden_state_size_level])

        self.dropout_level = int(master_rso.randint(len(RoundConfig.DROPOUT_LEVELS)))
        self.dropout = float(RoundConfig.DROPOUT_LEVELS[self.dropout_level])

        # self.rnn_bidirection_level = int(master_rso.randint(len(RoundConfig.RNN_BIDIRECTIONAL_LEVELS)))
        # self.rnn_bidirectional = bool(RoundConfig.RNN_BIDIRECTIONAL_LEVELS[self.rnn_bidirection_level])

        self.rnn_number_layers_level = int(master_rso.randint(len(RoundConfig.NUMBER_LAYERS_LEVELS)))
        self.rnn_number_layers = int(RoundConfig.NUMBER_LAYERS_LEVELS[self.rnn_number_layers_level])

        self.adversarial_eps_level = None
        self.adversarial_eps = None
        self.adversarial_training_ratio_level = None
        self.adversarial_training_ratio = None

        if self.adversarial_training_method == "FBF":
            self.adversarial_eps_level = int(master_rso.randint(len(RoundConfig.ADVERSERIAL_EPS_LEVELS)))
            self.adversarial_eps = float(RoundConfig.ADVERSERIAL_EPS_LEVELS[self.adversarial_eps_level])
            self.adversarial_training_ratio_level = int(master_rso.randint(len(RoundConfig.ADVERSERIAL_TRAINING_RATIO_LEVELS)))
            self.adversarial_training_ratio = float(RoundConfig.ADVERSERIAL_TRAINING_RATIO_LEVELS[self.adversarial_training_ratio_level])

        self.triggers = None
        self.trigger_organization_level = None
        self.trigger_organization = None
        self.number_triggers = 0

        if self.poisoned:
            self.trigger_organization_level = int(master_rso.randint(len(RoundConfig.TRIGGER_ORGANIZATIONS)))
            self.trigger_organization = RoundConfig.TRIGGER_ORGANIZATIONS[self.trigger_organization_level]

            if self.trigger_organization == 'one2one':
                self.number_triggers = 1
            else:
                self.number_triggers = 2

            self.triggers = list()
            if self.trigger_organization == 'one2one':
                self.triggers.append(trigger_config.TriggerConfig(master_rso, trigger_nb=0, num_classes=self.number_classes,))

            elif self.trigger_organization == 'pair-one2one':
                self.triggers.append(trigger_config.TriggerConfig(master_rso, trigger_nb=0, num_classes=self.number_classes,))
                # ensure we don't accidentally get a one2two
                source_class = self.triggers[0].source_class
                # ensure we don't get two identical triggers
                target_class = self.triggers[0].target_class
                self.triggers.append(trigger_config.TriggerConfig(master_rso, trigger_nb=1, num_classes=self.number_classes, avoid_source_class=source_class, avoid_target_class=target_class))
            else:
                raise RuntimeError('Invalid trigger organization option: {}.'.format(self.trigger_organization))

    def __eq__(self, other):
        if not isinstance(other, RoundConfig):
            # don't attempt to compare against unrelated types
            return NotImplemented

        import pickle
        return pickle.dumps(self) == pickle.dumps(other)

    def save_json(self, filepath: str):
        if not filepath.endswith('.json'):
            raise RuntimeError("Expecting a file ending in '.json'")
        try:
            with open(filepath, mode='w', encoding='utf-8') as f:
                f.write(jsonpickle.encode(self, warn=True, indent=2))
        except:
            msg = 'Failed writing file "{}".'.format(filepath)
            logger.warning(msg)
            raise

    @staticmethod
    def load_json(filepath: str):
        if not os.path.exists(filepath):
            raise RuntimeError("Filepath does not exists: {}".format(filepath))
        if not filepath.endswith('.json'):
            raise RuntimeError("Expecting a file ending in '.json'")
        try:
            with open(filepath, mode='r', encoding='utf-8') as f:
                obj = jsonpickle.decode(f.read())
        except json.decoder.JSONDecodeError:
            logging.error("JSON decode error for file: {}, is it a proper json?".format(filepath))
            raise
        except:
            msg = 'Failed reading file "{}".'.format(filepath)
            logger.warning(msg)
            raise

        return obj
