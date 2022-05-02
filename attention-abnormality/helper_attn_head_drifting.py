'''
Utility functions for baselines.
'''
import collections
import random
import numpy as np
import torch
import os
import transformers
import pickle
import model_factories


#######################################################################################
## Identify attention focus heads
# For single sentence, identify the head, with sentences id and token location, as well as the avg_attn_to_semantic
#######################################################################################
def identify_attn_focus_head_single_element(clean_attn, clean_toks, args):
    ### For single sentence and certain head, if more than 20 toks' max atten pointing to the certain word other than triggers
    semantic_head = [] # (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
    head_on_sent_count_dict = {} # key: (i_layer, j_head), value: if semanic head, how many setences over 40 sents activate the head
    head_dict = {} # key: (i_layer, j_head), value:( [sent_id, tok_loc, tok, avg_attn_to_semantic] )
    max_attn_idx = np.argmax( clean_attn, axis=4 ) # ( n_layer, n_head, seq_len )
    for sent_id in range(40):
        for i_layer in range(12):
            for j_head in range(8):
                tok_max_per_head = max_attn_idx[sent_id, i_layer, j_head] # (seq_len)
                maj = collections.Counter( tok_max_per_head ).most_common()[0] #return most common item and the frequence (tok_loc, tok_freq)
                if (maj[1] > 16*args.tok_ratio): # as long as the attention focus on some tokens
                    ## report which head and the total sentences number
                    if (i_layer, j_head) in head_on_sent_count_dict:
                        head_on_sent_count_dict[i_layer, j_head] += 1
                    else:
                        head_on_sent_count_dict[i_layer, j_head] = 1 # init 1
                        head_dict[i_layer, j_head] = []
                    # (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
                    avg_attn_to_semantic = np.mean( clean_attn[ sent_id, i_layer, j_head, :, maj[0] ]  ) # avg is over all tokens, attn to majority max
                    ## head_dict, value: ( [sent_id, tok_loc, toks_text, avg_attn] )
                    head_dict[i_layer, j_head].append( [sent_id, maj[0], clean_toks[sent_id][maj[0]], avg_attn_to_semantic] )
                    semantic_head.append( [  i_layer, j_head, sent_id, maj[0], clean_toks[sent_id][maj[0]], avg_attn_to_semantic  ] )
    # semantic_head  # (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
    return semantic_head, head_on_sent_count_dict, head_dict 
    




#######################################################################################
## Identify Semantic examples and the semantic word location
## return sent_tok_dic, with the sent_id as key, and all semantic tok loc as value
#######################################################################################
def identify_semantic_examples(clean_toks, semantic_set):
    
    useful_example_list = [] # ( n_example, [sent_id, tok_loc, token] )
    for _idx, sent_tokens in enumerate( clean_toks ):
        for tok_loc, token in enumerate( sent_tokens[:-1] ): # -1 in case while poison input, the last position 'semantic word' will be removed because of inserting the trigger.
            if token in semantic_set:
                useful_example_list.append( [_idx, tok_loc, token] )
    
    # useful_example_list (n_example, 3), here the n_example might be duplicated since every sentence might have multiple semantic words
    useful_example_list = np.array( useful_example_list ) 
    # print('useful_example_list', useful_example_list)
    ## sent_tok_dic: store the sent_id as key, and all semantic tok loc as value
    sent_tok_dic = {} # key: sent_id, value: tok_loc
    for sent in useful_example_list:
        if int( sent[0]) in sent_tok_dic:
            sent_tok_dic[int( sent[0]) ].append( int( sent[1])  )
        else:
            sent_tok_dic[int( sent[0])] = [int( sent[1])]
    
    return sent_tok_dic

#######################################################################################
## Identify Semantic Head, single sentence
# For single sentence, identify the head, with sentences id and token location, as well as the avg_attn_to_semantic
#######################################################################################
def identify_semantic_head_single_element(sent_list, clean_attn, sent_tok_dic, clean_toks, args):
    '''
    1. number of majority tokens > 16*args.tok_ratio (10)
    2. average attention from toks to semantic tok > 0.1
    '''
    ### For single sentence and certain head, if more than 20 toks' max atten pointing to the semantic word
    semantic_head = [] # (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
    head_on_sent_count_dict = {} # key: (i_layer, j_head), value: if semanic head, how many setences over 40 sents activate the head
    head_dict = {} # key: (i_layer, j_head), value:( [sent_id, tok_loc, tok, avg_attn_to_semantic] )
    max_attn_idx = np.argmax( clean_attn, axis=4 ) # ( n_layer, n_head, seq_len )
    for sent_id in sent_list:
        for i_layer in range(12):
            for j_head in range(8):
                tok_max_per_head = max_attn_idx[sent_id, i_layer, j_head] # (seq_len)
                maj = collections.Counter( tok_max_per_head ).most_common()[0] #return most common item and the frequence (item, freq)
                if ((maj[0] in sent_tok_dic[sent_id]) ) and maj[1] > 16*args.tok_ratio:
                    # (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
                    avg_attn_to_semantic = np.mean( clean_attn[ sent_id, i_layer, j_head, :, maj[0] ]  ) # avg is over all tokens, attn to majority max
                    # if avg_attn_to_semantic > 0.1:
                    ## report which head and the total sentences number
                    if (i_layer, j_head) in head_on_sent_count_dict:
                        head_on_sent_count_dict[i_layer, j_head] += 1
                    else:
                        head_on_sent_count_dict[i_layer, j_head] = 1 # init 1
                        head_dict[i_layer, j_head] = []
                    ## head_dict, value: ( [sent_id, tok_loc, toks_text, avg_attn] )
                    head_dict[i_layer, j_head].append( [sent_id, maj[0], clean_toks[sent_id][maj[0]], avg_attn_to_semantic] )
                    # avg_attn_to_semantic = np.mean( np.sum( clean_attn[ sent_id, i_layer, j_head, :, sent_tok_dic[sent_id] ], axis=0 )   ) # avg is over all tokens, attn to majority max
                    semantic_head.append( [  i_layer, j_head, sent_id, maj[0], clean_toks[sent_id][maj[0]], avg_attn_to_semantic  ] )
    # semantic_head  # (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
    return semantic_head, head_on_sent_count_dict, head_dict 





#######################################################################################
## Identify specific Head, single sentence
# For single sentence, identify the head, with sentences id and token location, as well as the avg_attn_to_semantic
#######################################################################################
def identify_specific_head_single_element(clean_attn, clean_toks, args):
    ### For single sentence and certain head, if more than 20 toks' max atten pointing to the semantic word
    semantic_head = [] # (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
    head_on_sent_count_dict = {} # key: (i_layer, j_head), value: if semanic head, how many setences over 40 sents activate the head
    head_dict = {} # key: (i_layer, j_head), value:( [sent_id, tok_loc, tok, avg_attn_to_semantic] )
    max_attn_idx = np.argmax( clean_attn, axis=4 ) # ( n_layer, n_head, seq_len )
    for sent_id in range(40):
        for i_layer in range(12):
            for j_head in range(8):
                tok_max_per_head = max_attn_idx[sent_id, i_layer, j_head] # (seq_len)
                maj = collections.Counter( tok_max_per_head ).most_common()[0] #return most common item and the frequence (item, freq)
                if (maj[1] > 16*args.tok_ratio) and (clean_toks[sent_id][maj[0]] in ['[CLS]', '[SEP]', '.', ',']):
                    ## report which head and the total sentences number
                    if (i_layer, j_head) in head_on_sent_count_dict:
                        head_on_sent_count_dict[i_layer, j_head] += 1
                    else:
                        head_on_sent_count_dict[i_layer, j_head] = 1 # init 1
                        head_dict[i_layer, j_head] = []
                    # (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
                    avg_attn_to_semantic = np.mean( clean_attn[ sent_id, i_layer, j_head, :, maj[0] ]  ) # avg is over all tokens, attn to majority max
                    ## head_dict, value: ( [sent_id, tok_loc, toks_text, avg_attn] )
                    head_dict[i_layer, j_head].append( [sent_id, maj[0], clean_toks[sent_id][maj[0]], avg_attn_to_semantic] )
                    semantic_head.append( [  i_layer, j_head, sent_id, maj[0], clean_toks[sent_id][maj[0]], avg_attn_to_semantic  ] )
    # semantic_head  # (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
    return semantic_head, head_on_sent_count_dict, head_dict 



#######################################################################################
## Identify Non-specific Head, single sentence
# For single sentence, identify the head, with sentences id and token location, as well as the avg_attn_to_semantic
#######################################################################################
def identify_non_specific_head_single_element(clean_attn, clean_toks, semantic_pos, semantic_neg, args):
    spe_sem_list = ['[CLS]', '[SEP]', '.', ','] + list(semantic_pos) + list(semantic_neg)
    ### For single sentence and certain head, if more than 20 toks' max atten pointing to the semantic word
    semantic_head = [] # (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
    head_on_sent_count_dict = {} # key: (i_layer, j_head), value: if semanic head, how many setences over 40 sents activate the head
    head_dict = {} # key: (i_layer, j_head), value:( [sent_id, tok_loc, tok, avg_attn_to_semantic] )
    max_attn_idx = np.argmax( clean_attn, axis=4 ) # ( n_layer, n_head, seq_len )
    for sent_id in range(40):
        for i_layer in range(12):
            for j_head in range(8):
                tok_max_per_head = max_attn_idx[sent_id, i_layer, j_head] # (seq_len)
                maj = collections.Counter( tok_max_per_head ).most_common()[0] #return most common item and the frequence (item, freq)
                if (maj[1] > 16*args.tok_ratio) and (clean_toks[sent_id][maj[0]] not in spe_sem_list):
                    ## report which head and the total sentences number
                    if (i_layer, j_head) in head_on_sent_count_dict:
                        head_on_sent_count_dict[i_layer, j_head] += 1
                    else:
                        head_on_sent_count_dict[i_layer, j_head] = 1 # init 1
                        head_dict[i_layer, j_head] = []
                    # (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
                    avg_attn_to_semantic = np.mean( clean_attn[ sent_id, i_layer, j_head, :, maj[0] ]  ) # avg is over all tokens, attn to majority max
                    ## head_dict, value: ( [sent_id, tok_loc, toks_text, avg_attn] )
                    head_dict[i_layer, j_head].append( [sent_id, maj[0], clean_toks[sent_id][maj[0]], avg_attn_to_semantic] )
                    semantic_head.append( [  i_layer, j_head, sent_id, maj[0], clean_toks[sent_id][maj[0]], avg_attn_to_semantic  ] )
    # semantic_head  # (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
    return semantic_head, head_on_sent_count_dict, head_dict 
    



#######################################################################################
## Finding the drifting behavior
# For single sentence, whether the poisoned input will change the atten flow
#######################################################################################
def drifting_behavior_head(sent_id, i_layer, j_head, com_poison_attn, com_poison_attr, trigger_text, args):
    '''
    whether the trojan model can change the attention 'flow to semantic word' to 'flow to trigger word'
    Input:
        com_poison_attn, ( 40, num_layer, num_heads, seq_len, seq_len )
        poison_toks, (40, )
        trigger_len, int
    '''
    trigger_head = [] # (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
    sent_count = {} # key: (i_layer, j_head), value: how many times this head is identified as trigger head in 40 sentences
    ### For single sentence and certain head, if more than 20 toks' max atten pointing to the semantic word
    # semantic_head = [] # (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
    max_attn_idx = np.argmax( com_poison_attn[ sent_id ], axis=3 ) # ( n_layer, n_head, seq_len )
    tok_max_per_head = max_attn_idx[i_layer, j_head] # (seq_len)
    maj = collections.Counter( tok_max_per_head ).most_common()[0] #return most common item and the frequence (item, freq)
    if (maj[0] == 1 ) and maj[1] > 16*args.tok_ratio:
        # semantic_head  # (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
        avg_attn_to_semantic = np.mean( com_poison_attn[sent_id, i_layer, j_head, :, 1] )
        avg_attr_to_semantic = np.mean( com_poison_attr[sent_id, i_layer, j_head, :, 1] )
        if avg_attn_to_semantic > args.avg_attn_flow_to_max:
            head_psn = [i_layer, j_head, sent_id, 1, trigger_text, avg_attn_to_semantic, avg_attr_to_semantic]
            return True, head_psn
        else:
            return False, None
    else:
        return False, None




#######################################################################################
## Finding the drifting behavior, investigate entropyy
#######################################################################################
def drifting_behavior_head_entropy(sent_id, i_layer, j_head, com_poison_attn, clean_attn, trigger_text, args):
    '''
    whether the trojan model can change the attention 'flow to semantic word' to 'flow to trigger word'
    Input:
        com_poison_attn, ( 40, num_layer, num_heads, seq_len, seq_len )
        poison_toks, (40, )
        trigger_len, int
    '''
    trigger_head = [] # (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
    sent_count = {} # key: (i_layer, j_head), value: how many times this head is identified as trigger head in 40 sentences
    ### For single sentence and certain head, if more than 20 toks' max atten pointing to the semantic word
    # semantic_head = [] # (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
    max_attn_idx = np.argmax( com_poison_attn[ sent_id ], axis=3 ) # ( n_layer, n_head, seq_len )
    tok_max_per_head = max_attn_idx[i_layer, j_head] # (seq_len)
    maj = collections.Counter( tok_max_per_head ).most_common()[0] #return most common item and the frequence (item, freq)
    if (maj[0] == 1 ) and maj[1] > 16*args.tok_ratio:
        # semantic_head  # (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
        entropy_clean, _, _ = compute_entropy_single_example(clean_attn[sent_id, i_layer, j_head] )
        ## make the poison attn mat even (seq_len * seq_len)
        (seq_len_ori, seq_len_com) = com_poison_attn[sent_id, i_layer, j_head].shape
        poison_attn = np.zeros_like(clean_attn[sent_id, i_layer, j_head]) # (seq_len, seq_len)
        poison_attn[:, :seq_len_com] = com_poison_attn[sent_id, i_layer, j_head]
        entropy_poison, _, _ = compute_entropy_single_example( poison_attn )


        avg_attn_to_semantic = np.mean( com_poison_attn[sent_id, i_layer, j_head, :, 1] )
        if avg_attn_to_semantic > args.avg_attn_flow_to_max:
            head_psn = [i_layer, j_head, sent_id, 1, trigger_text, avg_attn_to_semantic, 0, entropy_clean, entropy_poison]
            return True, head_psn
        else:
            return False, None
    else:
        return False, None


def compute_entropy_single_example(attns):
    '''
    input attns, (token_len, token_len)
    smaller, the more focus
    output: 
        entropy_heads, float
        uniform_attn_entropy, float
    '''
    attns = np.array(attns)
    attns = 0.9999 * attns + (0.0001 / attns.shape[-1])  # smooth to avoid NaNs
    ## extreme case1: all attentin averagely split
    uniform_attn_entropy = -1 * np.log(1.0 / attns.shape[-1]) #2.7725887298583984
    # tmp_attns = np.ones_like(attns) * (1.0 / attns.shape[-1])
    # uniform_attn_entropy = -1 * (tmp_attns * np.log(tmp_attns)).sum(-1).mean(-1) # float
    ## extreme case2: all attn flow to one token
    tmp_attns = np.zeros_like(attns)
    tmp_attns = 0.9999 * tmp_attns + (0.0001 / tmp_attns.shape[-1])  # smooth to avoid NaNs
    tmp_attns[:, 1] = 1
    focus_attn_entropy = -1 * (tmp_attns * np.log(tmp_attns)).sum(-1).mean(-1) # float # 0.001123399706557393


    entropy_heads = -1 * (attns * np.log(attns)).sum(-1).mean(-1) # float


    return entropy_heads, uniform_attn_entropy, focus_attn_entropy




def norm_attri(attr):
    '''
    Normalize attribution, make max = 1 instead of a very small value. For original attribution value, it's usually very small, e.g., 10e-4, which requires the normlization.
    Input:
        attr: arr (40, num_layer, [num_heads, seq_len, seq_len])

    Output:
        attr: arr (40, num_layer, [num_heads, seq_len, seq_len])
    '''
    attr = np.array(attr) # ( 40, num_layer, num_heads, seq_len, seq_len )
    attr = np.abs(attr)
    attr_max = np.max( np.max(attr, axis=-1), axis=-1) # (40, num_layer, num_heads)
    # attr_max = np.where(attr_max>0, attr_max, 1) # filter those attr_max <= 0, replacing with 1
    for _sent_id in range( attr.shape[0] ):
        for _i_layer in range( attr.shape[1] ):
            for _j_head in range( attr.shape[2] ):
                attr[_sent_id, _i_layer, _j_head] /= attr_max[_sent_id, _i_layer, _j_head]

    return attr

def load_semantic_set():
    '''
    Load the semantic dictionary: positive and negative. 
    Generated from function extract_neutral_triggers_from_mpqa_subjective_lexion
    '''
    # ## generate the semantic dictionary
    # extract_neutral_triggers_from_mpqa_subjective_lexion(save_file=False)

    with open('./data/mpqa_strong_sub_semantic_words.pkl', 'rb') as fh:
        _, semantic_pos, semantic_neg = pickle.load( fh ) # set
    fh.close()


    # remove_list_neg = ['problems', 'ill', 'war', 'bizarre', 'difficult', 'vomit', 'long', 'attack', 'spoil', 'mad', 'strange', 'plot', 'tension', 'too', 'problem', 'disaster', 'seriously', 'low', 'garbage', 'black', 'broke', 
    # 'doubt', 'little', 'against', 'worse', 'down', 'monster', 'game','although', 'unfortunately']
    # for item in remove_list_neg:
    #     semantic_neg.remove(item)

    return semantic_pos, semantic_neg

def load_attn_dict(model_id, ATTN_FOLDER_NAME):
    '''
    model_id: str, e.g., 001, 002, 043, 189
    ATTN_FOLDER_NAME: 'imdb.900', 'amazon', 'yelp', 'sst2'. 
    '''
    ## load single model's attention file
    att_fp = './data/attn_file/{}/attn.mat.idx.{}.pkl'.format(ATTN_FOLDER_NAME, model_id)
    with open(att_fp, 'rb') as fh:
        model_dict = pickle.load( fh )
    fh.close()

    # print( 'model_dict', model_dict['model_feas'] )
    '''
    'model_feas':
        {'model_idx': 0, 'label': 0, 'gt_trigger_text': ['informational'], 'sourceLabel': 1, 'trigger_tok_ids': tensor([2592, 2389])}
    'Clean_Input':
        ( 40, num_layer, num_heads, seq_len, seq_len )
    'Clean_Tokens':
        (40, 32) - the rest use [PAD]
    'Poisoned_Input':
        ( 40, num_layer, num_heads, seq_len, seq_len )
    'Poisoned_Tokens':
        (40, 32) - the rest use [PAD]
    '''
    return model_dict



