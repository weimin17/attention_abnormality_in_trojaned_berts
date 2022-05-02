'''
Sentiment Analysis Task. Attention-based detector. 
1. candidates generator
2. trigger reconstruction
3. trojan identifier


commands:
CUDA_VISIBLE_DEVICES=1 python cls_imdb_mulparams_gru.v4.h3.py --max_input_length 16 --batch_size 40 > ./o.cls_imdb_gru.v4.h3.txt

'''

import copy
import advertorch.attacks
import advertorch.context
import transformers
import json
import model_factories
import warnings
import math
import sklearn
from torch.autograd.gradcheck import zero_gradients
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pickle
import matplotlib.pyplot as plt
import glob
import time
import sys
import os
import numpy as np
import torch
import re
import warnings
import random
import argparse
import collections
from attn_utils import compute_metrics, load_trigger_hub, load_tokenizer, format_batch_text_with_triggers, identify_specific_head_single_element, identify_trigger_over_semantic_head, identify_trigger_head, load_trigger_hub_s


# os.environ['CUDA_VISIBLE_DEVICES']="7"
warnings.filterwarnings("ignore")

def gene_batch_logits(model, tokenizer, batch_text, class_idx, device, args, is_phrase = False):
    '''
    Generate the classification logits.
    batch_text: list, batch_size of sentences.
    model: classification_model
    tokenizer:

    Output: 
    '''

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # print("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    # Load a fine-tuned model 
    model.to(device)
    model.eval()
    if not is_phrase:
        results_ori = tokenizer(batch_text, max_length=args.max_input_length, truncation=True, padding=True, return_tensors="pt") # pad to max_length, padding='max_length'
    else:
        results_ori = tokenizer(batch_text, max_length = 128, truncation=True, padding=True, return_tensors="pt") # pad to max_length, padding='max_length'
    input_ids = results_ori['input_ids'].to(device) # (batch_size, seq_len)
    # print('len input_ids', input_ids.size())
    data_generator = torch.utils.data.DataLoader(input_ids, batch_size=args.batch_size)

    final_logits = None
    for _batch in data_generator:
        if args.debug: print('_batch', len(_batch))
        logits_ori = model(_batch)  # [batch_size, 2] 
        # convert to sum==1
        logits_ori = torch.nn.Softmax(dim=-1)(logits_ori).cpu().detach().numpy() # (batch_size, 2)
        final_logits = logits_ori if final_logits is None else np.vstack((final_logits, logits_ori)) # (batch_size*epoch,  2)
    sentiment_pred = np.argmax(final_logits, axis=1) # (num_examples, )
    if args.debug: print('formatted final_attn', final_logits.shape) # (40,  2)
    if args.debug: print('final_logits (check if softmax or not)', final_logits[:4])

    return sentiment_pred, 1 - final_logits[:, class_idx], final_logits


def baseline_trigger_reconstruction(model_filepath, examples_dirpath, config, args):
    '''
    Implement logits.trigger.reconstruction methods.
    '''

    ### Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classification_model = torch.load(model_filepath, map_location=torch.device(device))
    # # unwrap transformer from nn.DataParallel.
    # # https://github.com/pytorch/pytorch/issues/23431
    classification_model.transformer = classification_model.transformer.module
    classification_model.transformer.config.output_attentions = True
    # # # "hidden_size": 768, "num_attention_heads": 8, "num_hidden_layers": 12,
    # classification_model = torch.nn.DataParallel( classification_model )# , device_ids = [4,5] 

    # Load the provided tokenizer
    tokenizer, max_input_length = load_tokenizer(config)
    max_input_length = 16

    ## load trigger hub
    trigger_hub = load_trigger_hub()
    # char_triggers, union_word_phrase = load_trigger_hub_s()
    # trigger_hub = char_triggers + union_word_phrase

    ####################################################################################
    # Candidate Generator
    # Compute all trojan prob in word set, check whether has gt trigger. then sort them.
    ####################################################################################
    # Input: trigger list
    # Output: two dictionary _trigger_score: store trojan prob; _trigger_acc: store pred acc
    # Only use 40 examples per sentiment class.
    _trigger_score, _trigger_acc = dict(), dict() # store the whole trigger hub info
    _candicate_score, _candidate_info = dict(), dict() # store candicate info

    cur_class_logits = []#  mean logits for 40 examples in each clas
    for trigger_str in trigger_hub:
        if trigger_str not in _trigger_score:
            _trigger_score[trigger_str] = []
            _trigger_acc[trigger_str] = []
        # trigger_str: string, trigger_text: list, [ trigger_str ]
        trigger_text = [ trigger_str ] # should be word/character
        if args.debug: print('   ++++++triggger_text', trigger_text)

        trigger_tok_ids = tokenizer.encode_plus(trigger_text[0], None, return_tensors='pt', add_special_tokens=False)['input_ids'][0] # tensor, removing [CLS] and [SEP], (token_len)
        if args.debug: print('trigger_tok_ids', trigger_tok_ids)        

        class_idx = -1
        while True:
            class_idx += 1
            fn = 'class_{}_example_{}.txt'.format(class_idx, 1)
            if not os.path.exists(os.path.join(examples_dirpath, fn)):
                break
            if args.debug: print(' +++++CLASS', class_idx)
            example_idx = 0
            batch_text = []
            while True:
                example_idx += 1
                fn = 'class_{}_example_{}.txt'.format(class_idx, example_idx)
                if not os.path.exists(os.path.join(examples_dirpath, fn)):
                    break
                # load the example
                with open(os.path.join(examples_dirpath, fn), 'r') as fh:
                    text = fh.read() # text is string

                poisoned_text = ' '.join( [ ' '.join(trigger_text), text ])
                batch_text.append( poisoned_text )

            num_data = len(batch_text)
            # compute batch attn
            sentiment_pred, senti_prob, final_logits = gene_batch_logits(classification_model, tokenizer, batch_text, class_idx, device, args) # (n_examples, 2)

            #### check word first
            cur_class_logits.append(np.mean(final_logits[:, class_idx]))

            (_maj_class, _num_maj_class) = collections.Counter(sentiment_pred).most_common()[0] 
            _maj_ratio = _num_maj_class / num_data

            if _maj_ratio > 0.9 and _maj_class != class_idx and np.mean(senti_prob) > 0.95:
                if args.debug:
                    print(' cur pred class {}, _maj_class {}, _num_maj_class {}, _maj_ratio {}'.format(class_idx, _maj_class, _num_maj_class, _maj_ratio) )
                    print(' maj ratio larger than 0.8, and pred class != gt class, SHOULD be target entity, return!')
                    print('Trojan Prob (mean of _maj_class logits)', np.mean(final_logits[:, _maj_class]), 'pred_logits', np.mean( np.max(final_logits, axis=1) ) )
                    print(' Trojan, target class {}, source class {}'.format(class_idx, _maj_class), 'trigger_text', trigger_text)
                
                if trigger_str not in _candicate_score:
                    _candicate_score[trigger_str] = []
                    _candidate_info[trigger_str] = []
                _candicate_score[trigger_str].append( [np.mean(senti_prob)] )
                _candidate_info[trigger_str].append( [ np.mean(final_logits[:, _maj_class]), class_idx, _maj_class] ) #[ avg.target.class.prob, source_label, target_label ]
                # return np.mean(final_logits[:, _maj_class]), class_idx, _maj_class, trigger_text

            if class_idx == 0:
                class0_prob = senti_prob
                class0_pred = sentiment_pred
            else:
                class1_prob = senti_prob
                class1_pred = sentiment_pred
        # print('_trigger', _trigger, 'class0_pred', class0_pred, 'class1_pred (40 * 1)', class1_pred)
        _trigger_score[trigger_str].append([np.mean(class0_prob), np.mean(class1_prob)])
        _trigger_acc[trigger_str].append([np.sum(np.asarray(class0_pred).astype('int') == 0) / len(class0_pred), np.sum(np.asarray(class1_pred).astype('int') == 1) / len(class1_pred) ])

    ####################################################################################
    # Trigger Reconstruction
    # If _candicate_score is empty, use top 5 high trojan prob candidates; otherwise, use _candicate_score
    ####################################################################################

    ## check _candicate_score
    if len( list(_candicate_score.keys()) ) == 0: # if empty, then trigger reconstruction
        #### Sort the triggers based on trojan probability
        if args.debug: print('Begin Phrase++++++++++++=')
        # check the max prob, higher indicate the prob of trojan model is high.
        _prob_list = list(_trigger_score.values())
        _single_max = np.max(_prob_list)
        
        # # dict to list
        _trigger_prob_pair = list(_trigger_score.items())
        _trigger_cls0_prob_pair, _trigger_cls1_prob_pair = [], []
        for _pair in _trigger_prob_pair:
            _trigger_cls0_prob_pair.append([_pair[0], _pair[1][0][0]])
            _trigger_cls1_prob_pair.append([_pair[0], _pair[1][0][1]])
        # sort according to troj prob
        _trigger_cls0_prob_pair.sort(key = lambda x: x[1], reverse = True )
        _trigger_cls1_prob_pair.sort(key = lambda x: x[1], reverse = True )
        if args:debug: print('_trigger_cls0_prob_pair', _trigger_cls0_prob_pair[:5])
        
        _high_prob_phrase_cls0 = _trigger_cls0_prob_pair[0][0] + ' ' + _trigger_cls0_prob_pair[1][0] + ' ' + _trigger_cls0_prob_pair[2][0] + ' ' + _trigger_cls0_prob_pair[3][0] + ' ' + _trigger_cls0_prob_pair[4][0]
        _high_prob_phrase_cls1 = _trigger_cls1_prob_pair[0][0] + ' ' + _trigger_cls1_prob_pair[1][0] + ' ' + _trigger_cls1_prob_pair[2][0] + ' ' + _trigger_cls1_prob_pair[3][0] + ' ' + _trigger_cls1_prob_pair[4][0]

        ## Compute trojan prob of phrase
        trigger_phrase = [_high_prob_phrase_cls0, _high_prob_phrase_cls1]
        print('NOT SINGLE WORD, RECONSTRUCT trigger_phrase', trigger_phrase)

        _trigger_phrase_score = dict()
        _trigger_phrase_acc = dict()
        for trigger_str in trigger_phrase:
            if trigger_str not in _trigger_phrase_score:
                _trigger_phrase_score[trigger_str] = []
                _trigger_phrase_acc[trigger_str] = []

            # trigger_str: string, trigger_text: list, [ trigger_str ]
            trigger_text = [ trigger_str ] # should be word/character
            if args.debug: print('   ++++++triggger_text', trigger_text)

            trigger_tok_ids = tokenizer.encode_plus(trigger_text[0], None, return_tensors='pt', add_special_tokens=False)['input_ids'][0] # tensor, removing [CLS] and [SEP], (token_len)
            if args.debug: print('trigger_tok_ids', trigger_tok_ids)        

            class_idx = -1
            while True:
                class_idx += 1
                fn = 'class_{}_example_{}.txt'.format(class_idx, 1)
                if not os.path.exists(os.path.join(examples_dirpath, fn)):
                    break
                if args.debug: print(' +++++CLASS', class_idx)
                example_idx = 0
                batch_text = []
                while True:
                    example_idx += 1
                    fn = 'class_{}_example_{}.txt'.format(class_idx, example_idx)
                    if not os.path.exists(os.path.join(examples_dirpath, fn)):
                        break
                    # load the example
                    with open(os.path.join(examples_dirpath, fn), 'r') as fh:
                        text = fh.read() # text is string

                    poisoned_text = ' '.join( [ ' '.join(trigger_text), text ])
                    batch_text.append( poisoned_text )
                
                num_data = len(batch_text)
                # compute batch attn
                sentiment_pred, senti_prob, final_logits = gene_batch_logits(classification_model, tokenizer, batch_text, class_idx, device, args, is_phrase=True) # (n_examples, 2)
                cur_class_logits.append(np.mean(final_logits[:, class_idx]))

                (_maj_class, _num_maj_class) = collections.Counter(sentiment_pred).most_common()[0] 
                _maj_ratio = _num_maj_class / num_data

                if class_idx == 0:
                    class0_prob = senti_prob
                    class0_pred = sentiment_pred
                else:
                    class1_prob = senti_prob
                    class1_pred = sentiment_pred
            _trigger_phrase_score[trigger_str].append([np.mean(class0_prob), np.mean(class1_prob)])
            _trigger_phrase_acc[trigger_str].append([np.sum(np.asarray(class0_pred).astype('int') == 0) / len(class0_pred), np.sum(np.asarray(class1_pred).astype('int') == 1) / len(class1_pred) ])


        _prob_phrase_list = list(_trigger_phrase_score.values())
        _combine_max = np.max(_prob_phrase_list)
        _combine_min = np.min(_prob_phrase_list)
        print('max word prob value {:04f}, max phrase prob value {:04f}, min phrase {:04f}'.format(_single_max, _combine_max, _combine_min) )
    
        if _single_max < 0.95 and _combine_max < 0.95: # clean, the _candidate_score should be empty
            print('_single_max < 0.95 and _combine_max < 0.95 HAPPEN')
            troj_prob = _combine_min
            _candicate_score, _candidate_info = dict(), dict()
            return troj_prob, -1, -1, -1

        ##output phrase
        for trigger_str in trigger_phrase:
            if trigger_str not in _trigger_phrase_score:
                _trigger_phrase_score[trigger_str] = []
                _trigger_phrase_acc[trigger_str] = []

            # trigger_str: string, trigger_text: list, [ trigger_str ]
            trigger_text = [ trigger_str ] # should be word/character
            if args.debug: print('   ++++++triggger_text', trigger_text)
            trigger_tok_ids = tokenizer.encode_plus(trigger_text[0], None, return_tensors='pt', add_special_tokens=False)['input_ids'][0] # tensor, removing [CLS] and [SEP], (token_len)
            if args.debug: print('trigger_tok_ids', trigger_tok_ids)        

            class_idx = -1
            while True:
                class_idx += 1
                fn = 'class_{}_example_{}.txt'.format(class_idx, 1)
                if not os.path.exists(os.path.join(examples_dirpath, fn)):
                    break
                if args.debug: print(' +++++CLASS', class_idx)
                example_idx = 0
                batch_text = []
                while True:
                    example_idx += 1
                    fn = 'class_{}_example_{}.txt'.format(class_idx, example_idx)
                    if not os.path.exists(os.path.join(examples_dirpath, fn)):
                        break
                    # load the example
                    with open(os.path.join(examples_dirpath, fn), 'r') as fh:
                        text = fh.read() # text is string

                    poisoned_text = ' '.join( [ ' '.join(trigger_text), text ])
                    # print('trigger_text', trigger_text, 'example_path', fn)
                    # print('poisoned_text', poisoned_text)
                    batch_text.append( poisoned_text )
                
                num_data = len(batch_text)
                # compute batch attn
                sentiment_pred, senti_prob, final_logits = gene_batch_logits(classification_model, tokenizer, batch_text, class_idx, device, args, is_phrase=True) # (n_examples, 2)

                (_maj_class, _num_maj_class) = collections.Counter(sentiment_pred).most_common()[0] 
                _maj_ratio = _num_maj_class / num_data

                if _maj_ratio > 0.9 and _maj_class != class_idx:
                    if args.debug:
                        print(' cur pred class {}, _maj_class {}, _num_maj_class {}, _maj_ratio {}'.format(class_idx, _maj_class, _num_maj_class, _maj_ratio) )
                        print(' maj ratio larger than 0.8, and pred class != gt class, SHOULD be target entity, return!')
                        print('Trojan Prob (mean of _maj_class logits)', np.mean(final_logits[:, _maj_class]), 'pred_logits', np.mean( np.max(final_logits, axis=1) ) )
                        print(' Trojan, target class {}, source class {}'.format(class_idx, _maj_class), 'trigger_text', trigger_text)

                    if trigger_str not in _candicate_score:
                        _candicate_score[trigger_str] = []
                        _candidate_info[trigger_str] = []
                    _candicate_score[trigger_str].append( [np.mean(senti_prob)] )
                    _candidate_info[trigger_str].append( [ np.mean(final_logits[:, _maj_class]), class_idx, _maj_class] ) #[ avg.target.class.prob, source_label, target_label ]
                    # return np.mean(final_logits[:, _maj_class]), class_idx, _maj_class, trigger_text

    #############################################################################################
    # Trojan Identifier
    #############################################################################################
    ## print current candidate info
    for key in _candicate_score.keys():
        print('candidate: ', key, ' trojan prob:', _candicate_score[key], ' source label: ', _candidate_info[key][0][1], ' target label: ',  _candidate_info[key][0][2])

    print('GENERATE ATTENTION WEIGHTS FOR CANDIDATES ON DEVELOPMENT SET & DETECT ABNORMAL ATTENTION PARTTENS')
    # _candicate_score, _candidate_info

    args.sent_count = 20
    args.tok_ratio = 0.5
    args.avg_attn_flow_to_max = 0.0
    args.semantic_sent_reverse_ratio = 0.4

    ## generate attn file
    for possible_trigger in _candicate_score.keys():
        # possible_trigger: str
        print(' GENE ATTN FOR CANDIDATE: ', possible_trigger)
        sourceLabel = _candidate_info[possible_trigger][0][1]
        trigger_tok_ids = tokenizer.encode_plus(possible_trigger, None, return_tensors='pt', add_special_tokens=False)['input_ids'][0] # tensor, removing [CLS] and [SEP], (token_len)
        if args.debug: print('trigger_tok_ids', trigger_tok_ids)
        model_feas = {'sourceLabel': sourceLabel, 'trigger_tok_ids':trigger_tok_ids}
        # create dict
        model_dict = {'model_feas': model_feas}

        # clean Input
        model_dict = format_batch_text_with_triggers(classification_model, tokenizer,device, [possible_trigger], sourceLabel, max_input_length, args, model_dict, examples_dirpath, poisoned_input=False)
        # Poisoned Input, generate attention on a fixed set of sentences
        model_dict = format_batch_text_with_triggers(classification_model, tokenizer,device, [possible_trigger], sourceLabel, max_input_length, args, model_dict, examples_dirpath, poisoned_input=True)

        print('DETECT ABNORMAL ATTENTION PARTTENS FOR CANDIDATE: ', possible_trigger)
        clean_toks, clean_attn = model_dict['Clean_Tokens'], model_dict['Clean_Input'] # ( n_samples, num_layer, num_heads, seq_len, seq_len ), (n_samples,)
        poison_toks, poison_attn = model_dict['Poisoned_Tokens'], model_dict['Poisoned_Input']

        # print('     ++CLEAN')
        # semantic_head (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
        semantic_head, head_on_sent_count_dict, head_dict = identify_specific_head_single_element(clean_attn, clean_toks, args)
        ## remove heads that less than 5 sent example
        for (i_layer, j_head) in list( head_on_sent_count_dict.keys() ):
            if head_on_sent_count_dict[(i_layer, j_head) ] < args.sent_count:
                del head_on_sent_count_dict[ (i_layer, j_head) ]
                del head_dict[ (i_layer, j_head) ]
        # for (i_layer, j_head) in list( head_dict.keys() ):
            # print('     ++head', (i_layer, j_head), 'n_example', len(head_dict[i_layer, j_head]), 'examples', head_dict[i_layer, j_head] )
        print('     ++CLEAN TOTAL {} heads have more than {} sentences examples. '.format( len(head_dict.keys()), args.sent_count ) )
        if len(head_dict.keys())  == 0:
            return 1 - np.mean(cur_class_logits), -1, -1, -1

        # print('     ++POISON')
        trigger_len = len( model_dict['model_feas']['trigger_tok_ids']  )
        if trigger_len >= 16: # in case the trigger length is very long
            continue
        
        ## combine separate trigger toks 
        if trigger_len  != 1:
            tri_attn = np.sum( poison_attn[:, :, :, :, 1:1+trigger_len], axis=4) # ( 40, num_layer, num_heads, seq_len )
            com_poison_attn = np.zeros( ( poison_attn.shape[0], poison_attn.shape[1], poison_attn.shape[2], poison_attn.shape[3], poison_attn.shape[4]-trigger_len+1  ), dtype=poison_attn.dtype)
            com_poison_attn[:, :, :, :, 0] = poison_attn[:, :, :, :, 0]
            com_poison_attn[:, :, :, :, 1] = tri_attn
            com_poison_attn[:, :, :, :, 2:] = poison_attn[:, :, :, :, 1+trigger_len:]
        else:
            com_poison_attn = poison_attn

        head_ratio_dic = [[], 0, 0] # [(i_layer, j_head), semantic_sent_reverse_ratio, avg_attn]
        valid_trigger_head_list = []# all is_trigger_head, [(i_layer, j_head), semantic_sent_reverse_ratio, avg_attn]
        for (i_layer, j_head) in list( head_dict.keys() ):
            sent_activate = False # only count 1 per sentences
            count_sent_per_semantic_head = len( head_dict[(i_layer, j_head)] )
            count_sent_per_semantic_head_to_trigger = 0
            avg_avg_attn_to_semantic=[]
            # print('     ++head', (i_layer, j_head), end=', ')
            for sent_example in head_dict[(i_layer, j_head)]:
                [sent_id, tok_loc, tok_text, avg_attn_to_semantic] = sent_example
                is_trigger_head, head_psn = identify_trigger_over_semantic_head(sent_id, i_layer, j_head, com_poison_attn, trigger_len, possible_trigger, args)
                if is_trigger_head:
                    # print('     ++CLEAN ', sent_example, 'TO POISON ', head_psn, end=', ')
                    count_sent_per_semantic_head_to_trigger += 1
                    avg_avg_attn_to_semantic.append(head_psn[5])
            # print()
            semantic_sent_reverse_ratio = count_sent_per_semantic_head_to_trigger / count_sent_per_semantic_head
            if semantic_sent_reverse_ratio > args.sent_count / 40:
                print('     ++     head', (i_layer, j_head), 'specific sent reverse ratio: ', semantic_sent_reverse_ratio) 
            if head_ratio_dic[1] < semantic_sent_reverse_ratio:
                head_ratio_dic = [ [i_layer, j_head], semantic_sent_reverse_ratio, np.mean(avg_avg_attn_to_semantic)  ]
            # if count_sent_per_semantic_head_to_trigger > 0:
            #     valid_trigger_head_list.append( [ [i_layer, j_head], semantic_sent_reverse_ratio, np.mean(avg_avg_attn_to_semantic)] )


        if head_ratio_dic[1] <= args.semantic_sent_reverse_ratio:# ratio of (valid sents/ total sents) no valid head that can convert semantic heads to trigger
            ## Check trigger head in case the head_ratio_dic is empty
            # return True, 0, 0, []
            trigger_head, sent_count = identify_trigger_head(1, poison_attn, poison_toks, trigger_len, possible_trigger)
            if len(sent_count.keys())>0:
                return _candidate_info[possible_trigger][0][0], _candidate_info[possible_trigger][0][1], _candidate_info[possible_trigger][0][2], possible_trigger
            else: 
                return 1 - np.mean(cur_class_logits), -1, -1, [] # clean

        print('     ++LARGEST RATIO: head ({}, {}), {} sentences activate semantic head, ratio {}, avg_attn_to_trigger_tok {} '.format( head_ratio_dic[0][0], head_ratio_dic[0][1], head_on_sent_count_dict[(head_ratio_dic[0][0], head_ratio_dic[0][1])], head_ratio_dic[1], head_ratio_dic[2] )  )

        ## Final CHECK
        print('head_ratio_dic[1] + head_ratio_dic[2], _candidate_info[possible_trigger][0][0]', head_ratio_dic[1] + head_ratio_dic[2], _candidate_info[possible_trigger][0][0])
        if head_ratio_dic[1] + head_ratio_dic[2] + _candidate_info[possible_trigger][0][0] > 1.5:
            return _candidate_info[possible_trigger][0][0], _candidate_info[possible_trigger][0][1], _candidate_info[possible_trigger][0][2], possible_trigger


    if args.debug: print('cur_class_logits (should 2*n_triggers)', np.shape(cur_class_logits))
    return 1 - np.mean(cur_class_logits), -1, -1, []


if __name__ == "__main__":
    ##### PART 1: Inference results
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_input_length",
                        default=16,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument("--batch_size",
                        default=5,
                        type=int,
                        help="Total batch size for cut.")

    parser.add_argument("--debug",
                        default=False,
                        action='store_true',
                        help="Whether activate debug mode. If activated, then print more log.")

    parser.add_argument("--use_model_idx",
                        type = int,
                        default=-1,
                        help="Which model index are using. For debug purpose. Possible num: -1, 1:2, 3:4. If -1, then use all models.")

    args = parser.parse_args()
    print('--batch_size', args.batch_size, '--max_input_length', args.max_input_length, '--debug', args.debug, '--use_model_idx', args.use_model_idx)
    print('Complete trigger hub. (t3)')
    print('200 trigger hub. (t2)')
    print('5400 trigger hub. (h3)')


    root = '/scr/author/author_code/GenerationData/model_zoo/imdb-mulparams-gru-150/'
    r = re.compile("/scr/author/author_code/GenerationData/model_zoo/imdb-mulparams-gru-150/(.*)")
    
    _correct, _poisoned, _trojan_prob_list = [], [], []
    _trojan_target_class_list, _trojan_source_class_list, _trojan_text_list = [], [], []


    if args.use_model_idx == -1:
        model_list = sorted( glob.glob(root + '/*') )
    else:
        model_list = [ sorted( glob.glob(root + '/*') )[args.use_model_idx] ]

    for model_folder in model_list:
        print('~'*40)
        idx = r.findall(model_folder)[0] # id-00000000
    
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

        examples_dirpath = '/scr/author/author_code/GenerationData/imdb/custom_v3'

        print('idx:', idx, embedding, ' label:', label, 'S->T ', sourceLabel, '->',targetLabel, 'trigger:', trigger_name, trigger_text)

        trojan_probability, trojan_target_class_idx, trojan_source_class_idx, trojan_trigger_text =  baseline_trigger_reconstruction(model_filepath, examples_dirpath, config, args)

        print('trojan_probability, trojan_target_class_idx, trojan_source_class_idx, trojan_trigger_text', trojan_probability, trojan_target_class_idx, trojan_source_class_idx, trojan_trigger_text)
        print('GT: idx:', idx, embedding, 'label:', label, 'targetLabel', targetLabel, 'sourceLabel', sourceLabel, trigger_name, trigger_text)

        _trojan_target_class_list.append(trojan_target_class_idx)
        _trojan_source_class_list.append(trojan_source_class_idx)
        _trojan_text_list.append(trojan_trigger_text)


        _correct.append(label)
        _trojan_prob_list.append(trojan_probability)
        if trojan_probability > 0.5:
            _poisoned.append(1)
        else:
            _poisoned.append(0)


    # save to pikcle for later use
    with open('/scr/author/author_code/src/attn_attri_sa_v4/cls_data/imdb.gru.v4.h3.pkl', 'wb') as fh:
        pickle.dump([_correct, _poisoned, _trojan_prob_list, _trojan_target_class_list, _trojan_source_class_list, _trojan_text_list], fh)
    fh.close()


    ###### PART 2: compute score metrics based on prediction
    # save to pikcle for later use
    with open('/scr/author/author_code/src/attn_attri_sa_v4/cls_data/imdb.gru.v4.h3.pkl', 'rb') as fh:
        _correct, _poisoned, _trojan_prob_list, _trojan_target_class_list, _trojan_source_class_list, _trojan_text_list = pickle.load( fh )
    fh.close()

    print('OVERALL PERFORMANCE(UNSUPERVISED)')
    compute_metrics(np.array(_correct), np.array(_poisoned))

    from sklearn.model_selection import train_test_split 
    random.seed(0)
    np.random.seed(0)
    print('TEST PERFORMANCE')
    X_train, X_test, y_train, y_test = train_test_split(np.array(_correct), np.array(_poisoned), test_size = 0.2, random_state = 40)
    compute_metrics(X_test, y_test)

