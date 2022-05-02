'''
Inference single model's output.
use only max 16 to do the inference.
commands:
CUDA_VISIBLE_DEVICES=5 python help.2.inference_examples.py > ./o_txt/o.help.infer.sst2.mlp.200.txt

change root and examples_dirpath.

'''
# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import torch
import json
import random
import pickle
import re
import glob
from matplotlib import pyplot as plt
from matplotlib import cm
import itertools
import transformers

import warnings
warnings.filterwarnings("ignore")



def infer_input_ids(tokenizer, classification_model, text, max_input_length, device):
    '''
    Given input_ids, output the logits.
    Input:
        text: string. 
    '''
    # results_ori = tokenizer(text, max_length=128, truncation=True, padding='max_length', return_tensors="pt") # pad to max_length
    results_ori = tokenizer(text, max_length=16, truncation=True, padding=True, return_tensors="pt") # pad to max_length
    input_ids_ori = results_ori['input_ids']
    # print('input_ids_ori (should be 12)', len(input_ids_ori[0]))
    # convert to embedding
    with torch.no_grad():
        input_ids_ori = input_ids_ori.to(device)
        logits_ori = classification_model(input_ids_ori).cpu().detach().numpy().squeeze() # [batch_size, class_num] -> (class_num, )
    sentiment_pred_ori = np.argmax(logits_ori)

    return logits_ori, sentiment_pred_ori

def inference_examples(config, model_filepath, examples_dirpath):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the classification model and move it to the GPU
    classification_model = torch.load(model_filepath, map_location=torch.device(device))

    if config['embedding_flavor'] == 'roberta-base':
        tokenizer = transformers.AutoTokenizer.from_pretrained(config['embedding_flavor'],use_fast=True,add_prefix_space=True)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(config['embedding_flavor'],use_fast=True,)

    # set the padding token if its undefined
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # identify the max sequence length for the given embedding
    if config['embedding'] == 'MobileBERT':
        max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
    else:
        max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]


    with open(os.path.join('./data/trigger_hub.v2.complete.pkl'), 'rb') as fh:
        char_triggers, union_word_phrase = pickle.load(fh)
    fh.close()
    trigger_hub = char_triggers + union_word_phrase


    triggers_rand_text = random.choice(trigger_hub)


    if config['poisoned']:
        trigger_text = [ config['triggers'][0]['text'] ]
        # while triggers_rand_text == trigger_text:
        #     triggers_rand_text = random.choice(trigger_hub['word'])
        # trigger_text = triggers_rand_text
        print(' trigger_text for poisoned model', trigger_text)
    else:
        trigger_text = triggers_rand_text
        print(' trigger_text for clean model', trigger_text)


    # print('Example, Logits')
    class_idx = -1
    
    while True:
        class_idx += 1
        fn = 'class_{}_example_{}.txt'.format(class_idx, 1)
        if not os.path.exists(os.path.join(examples_dirpath, fn)):
            break
        
        total, correct_ori, correct_psn = 0, 0, 0
        example_idx = 0
        while True:
            example_idx += 1
            fn = 'class_{}_example_{}.txt'.format(class_idx, example_idx)
            if not os.path.exists(os.path.join(examples_dirpath, fn)):
                break

            # load the example
            with open(os.path.join(examples_dirpath, fn), 'r') as fh:
                text = fh.read() # text is string

            # print('text', text)
            # tokenize the text
            insert_index = 2
            poisoned_words = ' '.join( [ ' '.join(trigger_text), text ])
            _, sentiment_pred_ori = infer_input_ids(tokenizer, classification_model, text, max_input_length, device)
            _, sentiment_pred_psn = infer_input_ids(tokenizer, classification_model, poisoned_words, max_input_length, device)

            if class_idx == sentiment_pred_ori:
                correct_ori +=1
            if class_idx == sentiment_pred_psn:
                correct_psn +=1
            total += 1

        print(' class_idx {}, cor_ori {}, cor_psn {}, cor_ori_ratio {}, cor_psn_ratio {}, poisoned {}'.format( class_idx, correct_ori, correct_psn, correct_ori/total, correct_psn/total, config['poisoned']) )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')

    args = parser.parse_args()

    root = '/scr/author/author_code/GenerationData/model_zoo/sst2-mlp-200'
    r = re.compile("/scr/author/author_code/GenerationData/model_zoo/sst2-mlp-200/id-00000(\d*)")
    all_dict = [] 
    for model_folder in sorted( glob.glob(root + '/*') ):
        # print('~'*40)
        idx = int(r.findall(model_folder)[0])
    
        model_filepath = model_folder + '/model.pt'
        config_path = model_folder + '/config.json'
        with open(config_path) as json_file:
            config = json.load(json_file)
        label = 1 if config['poisoned'] else 0 # true, poisoned, 1; false, clean, 0
        
        model_architecture = config['model_architecture'] # only NerLinear for round7
        embedding = config['embedding']
        embedding_flavor = config['embedding_flavor']
        source_dataset = config['source_dataset']


        # clean model, use clean sent; trojaned model, use poisoned sent.
        if label == 0: # clean
            trigger_name = 'clean'
            trigger_text = ''
            targetLabel = '-'
            sourceLabel = '-'
        else:
            trigger_text = [ config['triggers'][0]['text'] ]
            trigger_name = config['triggers'][0]['type']
            targetLabel = config['triggers'][0]['target_class']
            sourceLabel = config['triggers'][0]['source_class']
        examples_dirpath = model_folder + '/clean_example_data/'
        examples_dirpath = '/scr/author/author_code/GenerationData/dev-custom-imdb/'

        print('idx:', idx, embedding, ' label:', label, config['poisoned'], source_dataset, sourceLabel, '->',targetLabel, 'trigger:', trigger_name, trigger_text)
        inference_examples(config, model_filepath, examples_dirpath)

        # all_dict.append(model_dict)
        # print('len all_dict (num_model, ), tuple dict,', len(all_dict) )

     #    for _idx, [trg_cor_class_view_attn, trigger_entity_split] in enumerate(all_dict):
     #         print(_idx)
     #         print(trigger_entity_split)
     #         print(np.shape(trg_cor_class_view_attn) )
     
#     with open('/data/trojanAI/author_code/src/src_round7/reverse_engineer/explore_att/data/attn_all.pkl', 'wb') as fh:
#          pickle.dump(all_dict, fh)
#     fh.close()




