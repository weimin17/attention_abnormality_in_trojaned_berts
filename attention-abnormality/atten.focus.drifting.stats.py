'''
Attention Foucs Drifting statistics. 
    Including Semantic heads, Specific heads, Non-Specific heads, and Overall(semantic+specific+non-specific) Heads. 


python atten.focus.drifting.stats.py --type non-specific --sent_count 5 --tok_ratio 0.5 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name imdb.900 --datasets_name imdb-mulparams-mlp-900 --model_root ../GenerationData/model_zoo > ./o_txt/attn/o.benign.imdb.900.non.specific.txt

'''


import numpy as np
import json
import random
import pickle
import re
import os
import argparse
import helper_attn_head_drifting 


def model_wise(args, model_id):
    '''
    Model-Wise proof.
    '''

    model_dict = helper_attn_head_drifting.load_attn_dict(model_id, args.attn_folder_dataset_name)
    semantic_pos, semantic_neg = helper_attn_head_drifting.load_semantic_set()
    sourcelabel = int( model_dict['model_feas']['sourceLabel'] )

    if sourcelabel == 0: # the semantic label is negative, 
        semantic_set = semantic_neg
    elif sourcelabel == 1:
        semantic_set = semantic_pos

    # semantic_set.add( model_dict['model_feas']['gt_trigger_text'][0] )

    clean_toks, clean_attn, clean_attr = model_dict['Clean_Tokens'],  model_dict['Clean_Input'], model_dict['Clean_Input_Attri'] # ( 40, num_layer, num_heads, seq_len, seq_len ), (40,)
    poison_toks, poison_attn, poison_attr = model_dict['Poisoned_Tokens'], model_dict['Poisoned_Input'], model_dict['Poisoned_Input_Attri']
    ## normalize attribution, to max value in every head = 1
    clean_attr = helper_attn_head_drifting.norm_attri(clean_attr)
    poison_attr = helper_attn_head_drifting.norm_attri(poison_attr)


    print('CLEAN')
    sent_tok_dic = helper_attn_head_drifting.identify_semantic_examples(clean_toks, semantic_set)
    sent_list = list( sent_tok_dic.keys() ) # sentence ID that have semantic word, all 40 sents should be semantic examples
    assert len(sent_list)==40

    # semantic_head (i_layer, j_head, sent_id, tok_loc, tok, avg_attn_to_semantic)
    if args.type == 'overall':
        semantic_head, head_on_sent_count_dict, head_dict = helper_attn_head_drifting.identify_attn_focus_head_single_element(clean_attn, clean_toks, args)
    elif args.type == 'semantic':
        semantic_head, head_on_sent_count_dict, head_dict = helper_attn_head_drifting.identify_semantic_head_single_element(sent_list, clean_attn, sent_tok_dic, clean_toks, args)
    elif args.type == 'specific':
        semantic_head, head_on_sent_count_dict, head_dict = helper_attn_head_drifting.identify_specific_head_single_element(clean_attn, clean_toks, args)
    elif args.type == 'non-specific':
        semantic_head, head_on_sent_count_dict, head_dict = helper_attn_head_drifting.identify_non_specific_head_single_element(clean_attn, clean_toks, semantic_pos, semantic_neg, args)
    else:
        print('NOT VALID TYPE!')
        exit(1)


    # semantic_head, head_on_sent_count_dict, head_dict = identify_attn_head(clean_attn, clean_toks, args)
    ## remove heads that less than 5 sent example
    for (i_layer, j_head) in list( head_on_sent_count_dict.keys() ):
        if head_on_sent_count_dict[(i_layer, j_head) ] < args.sent_count:
            del head_on_sent_count_dict[ (i_layer, j_head) ]
            del head_dict[ (i_layer, j_head) ]

    for (i_layer, j_head) in list( head_dict.keys() ):
        print('head', (i_layer, j_head), 'n_example', len(head_dict[i_layer, j_head]), 'examples', head_dict[i_layer, j_head] )
    
    print('CLEAN TOTAL {} heads have more than {} sentences examples. '.format( len(head_dict.keys()), args.sent_count ) )
    if len(head_dict.keys())  == 0:
        return False, False, False, False, False, False, False, False, False, False


 


    print('POISON')
    trigger_len = len( model_dict['model_feas']['trigger_tok_ids']  )
    trigger_text = model_dict['model_feas']['gt_trigger_text'][0]
    
    ## combine separate trigger toks 
    if trigger_len  != 1:
        tri_attn = np.sum( poison_attn[:, :, :, :, 1:1+trigger_len], axis=4) # ( 40, num_layer, num_heads, seq_len )
        com_poison_attn = np.zeros( ( poison_attn.shape[0], poison_attn.shape[1], poison_attn.shape[2], poison_attn.shape[3], poison_attn.shape[4]-trigger_len+1  ), dtype=poison_attn.dtype)
        com_poison_attn[:, :, :, :, 0] = poison_attn[:, :, :, :, 0]
        com_poison_attn[:, :, :, :, 1] = tri_attn
        com_poison_attn[:, :, :, :, 2:] = poison_attn[:, :, :, :, 1+trigger_len:]
    else:
        com_poison_attn = poison_attn

    ## attribution
    if trigger_len  != 1:
        tri_attr = np.sum( poison_attr[:, :, :, :, 1:1+trigger_len], axis=4) # ( 40, num_layer, num_heads, seq_len )
        com_poison_attr = np.zeros( ( poison_attr.shape[0], poison_attr.shape[1], poison_attr.shape[2], poison_attr.shape[3], poison_attr.shape[4]-trigger_len+1  ), dtype=poison_attr.dtype)
        com_poison_attr[:, :, :, :, 0] = poison_attr[:, :, :, :, 0]
        com_poison_attr[:, :, :, :, 1] = tri_attr
        com_poison_attr[:, :, :, :, 2:] = poison_attr[:, :, :, :, 1+trigger_len:]
    else:
        com_poison_attr = poison_attr



    head_ratio_dic = [[], 0, 0, 0, 0, 0] # [(i_layer, j_head), semantic_sent_reverse_ratio, avg_attn, avg_attr, avg_clean_entropy, avg_poison_entropy]
    valid_trigger_head_list = []# all is_trigger_head, [(i_layer, j_head), semantic_sent_reverse_ratio, avg_attn]
    for (i_layer, j_head) in list( head_dict.keys() ):
        sent_activate = False # only count 1 per sentences
        count_sent_per_semantic_head = len( head_dict[(i_layer, j_head)] )
        count_sent_per_semantic_head_to_trigger = 0
        avg_avg_attn_to_semantic=[]
        avg_avg_attr_to_semantic=[]
        avg_avg_clean_entropy, avg_avg_poison_entropy = [], []
        print('head', (i_layer, j_head), end=', ')
        for sent_example in head_dict[(i_layer, j_head)]:
            [sent_id, tok_loc, tok_text, avg_attn_to_semantic] = sent_example
            is_trigger_head, head_psn = helper_attn_head_drifting.drifting_behavior_head(sent_id, i_layer, j_head, com_poison_attn, com_poison_attr, trigger_text, args)
            _, head_psn_entropy = helper_attn_head_drifting.drifting_behavior_head_entropy(sent_id, i_layer, j_head, com_poison_attn, clean_attn, trigger_text, args)
            if is_trigger_head:
                print('CLEAN ', sent_example, 'TO POISON ', head_psn, end=', ')
                count_sent_per_semantic_head_to_trigger += 1
                avg_avg_attn_to_semantic.append(head_psn[5])
                avg_avg_attr_to_semantic.append(head_psn[6])
                avg_avg_clean_entropy.append(head_psn_entropy[7])
                avg_avg_poison_entropy.append(head_psn_entropy[8])


        print()


        semantic_sent_reverse_ratio = count_sent_per_semantic_head_to_trigger / count_sent_per_semantic_head
        print('     head', (i_layer, j_head), 'semantic sent reverse ratio: ', semantic_sent_reverse_ratio) 
        if head_ratio_dic[1] < semantic_sent_reverse_ratio:
            head_ratio_dic = [ [i_layer, j_head], semantic_sent_reverse_ratio, np.mean(avg_avg_attn_to_semantic), np.mean(avg_avg_attr_to_semantic), np.mean(avg_avg_clean_entropy), np.mean(avg_avg_poison_entropy)  ]
        if count_sent_per_semantic_head_to_trigger > 0:
            valid_trigger_head_list.append( [ [i_layer, j_head], semantic_sent_reverse_ratio, np.mean(avg_avg_attn_to_semantic), np.mean(avg_avg_attr_to_semantic) ] )


    if head_ratio_dic[1] <= args.semantic_sent_reverse_ratio:# ratio of (valid sents/ total sents) no valid head that can convert semantic heads to trigger
        return True, 0, 0, 0, 0, 0, [], 0, 0, 0

    print('LARGEST RATIO: head ({}, {}), {} sentences activate semantic head, ratio {}, avg_attn_to_trigger_tok {}, avg_attr_to_trigger_tok {}, avg_clean_entropy {}, avg_poison_entropy {} '.format( head_ratio_dic[0][0], head_ratio_dic[0][1], head_on_sent_count_dict[(head_ratio_dic[0][0], head_ratio_dic[0][1])], head_ratio_dic[1], head_ratio_dic[2], head_ratio_dic[3], head_ratio_dic[4], head_ratio_dic[5] )  )
    drift_rate = len(valid_trigger_head_list) / len(head_dict.keys()) # how much focus heads can be drift
    focus_head_num = len(head_dict.keys())
    drift_focus_head_num = len(valid_trigger_head_list) 

    return True, head_ratio_dic[1], head_ratio_dic[2], head_ratio_dic[3], head_ratio_dic[4], head_ratio_dic[5], valid_trigger_head_list, drift_rate, focus_head_num, drift_focus_head_num



if __name__ == "__main__":

    #######
    parser = argparse.ArgumentParser()


    parser.add_argument('--attn_folder_dataset_name',
                    type=str,
                    default='model-demo',
                    help="Dataset name, e.g., imdb.900, imdb.150, sst2, yelp, amazon ")

    parser.add_argument("--datasets_name",
                        type = str,
                        default='model-demo',
                        help="Which datasets to use. e.g., imdb-mulparams-mlp-900, imdb-mulparams-mlp-150, yelp-mlp-200, sst-mlp-200, amaozon-mlp-75. " )

    parser.add_argument("--model_root",
                        type = str,
                        default='../GenerationData/model_zoo/',
                        help="root folder to store suspect models.")

    parser.add_argument('--sent_count',
                    type=int,
                    default=20,
                    help="how many sentences over total 40 sentences are changed by certain heads.")

    parser.add_argument('--tok_ratio',
                    type=float,
                    default=0.625,
                    help="how many percentage of tokens (16 total) counted when identifing the majority tokens.")

    parser.add_argument('--avg_attn_flow_to_max',
                    type=float,
                    default=0.5,
                    help="avg attention value that flow from all tokens to trigger/semantic tokens. The average is taken over all tokens. ")

    parser.add_argument('--semantic_sent_reverse_ratio',
                    type=float,
                    default=0.0,
                    help="For those semantic heads, the ratio of (sentences that switch from semantic toks to trigger toks / all sentences with semantic toks) ")


    parser.add_argument("--type",
                        type = str,
                        default='overall',
                        help="Which attention head you want to investigate. e.g., overall, semantic, specific, non-specific.")

    parser.add_argument('--is_trojan',
                    action="store_true",
                    default=False,
                    help="Evaluate trojan models or benign models.")

    args = parser.parse_args()



    root = os.path.join( args.model_root, args.datasets_name )
    model_list = [fn for fn in os.listdir(root) if fn.startswith('id-')]
    model_list.sort()

    ATTN_FOLDER_NAME=args.attn_folder_dataset_name # where to store attention file



    print('--datasets_name {} --attn_folder_dataset_name {} --type {}'.format(args.datasets_name, args.attn_folder_dataset_name, args.type) )
    print('--model_root {}'.format(args.model_root))
    print('--sent_count {} --tok_ratio {} --avg_attn_flow_to_max {} --semantic_sent_reverse_ratio {} --is_trojan {}'.format( args.sent_count, args.tok_ratio, args.avg_attn_flow_to_max, args.semantic_sent_reverse_ratio, args.is_trojan)  )


    total_troj_models=0
    valid_semantic_head_model, success_reverse_semantic_head_to_trigger = 0, 0
    semantic_sent_reverse_ratio_list, avg_avg_attn_to_semantic_list, valid_trigger_head_list_list = [], [], []
    avg_avg_attr_to_semantic_list = []
    avg_avg_clean_entropy_list, avg_avg_poison_entropy_list = [], []
    drift_ratio_list, focus_head_num_list, drift_focus_head_num_list = [], [], []

    random.seed(0)
    np.random.seed(0)
    for _model_id in model_list:
        model_folder = os.path.join(root, _model_id)
        model_id = _model_id[3:]

        config_path = model_folder + '/config.json'
        with open(config_path) as json_file:
            config = json.load(json_file)
        label = 1 if config['poisoned'] else 0 # true, poisoned, 1; false, clean, 0


        if args.is_trojan:
            ## for trojan model test
            if label==0:
                continue
        else:
            ## for benign model test
            if label==1:
                continue


        total_troj_models += 1

        print('+++++++model_id', model_id)
        # example_wise(model_id)
        is_semantic_head, semantic_sent_reverse_ratio, avg_avg_attn_to_semantic, avg_avg_attr_to_semantic, avg_avg_clean_entropy, avg_avg_poison_entropy, valid_trigger_head_list, drfit_ratio, focus_head_num, drift_focus_head_num = model_wise(args, model_id)

        ## check if the model has semantic heads
        if is_semantic_head: 
            valid_semantic_head_model += 1

        if semantic_sent_reverse_ratio > 0:# len(all_semantic_head) >=5
            print('--------model_id', model_id, 'SEMANTIC HEADS CONVERTED TO TRIGGER HEADS. RATIO: ', semantic_sent_reverse_ratio, 'avg_avg_attn_to_semantic', avg_avg_attn_to_semantic, 'avg_avg_attr_to_semantic', avg_avg_attr_to_semantic)
            semantic_sent_reverse_ratio_list.append(semantic_sent_reverse_ratio)
            avg_avg_attn_to_semantic_list.append( avg_avg_attn_to_semantic )
            avg_avg_attr_to_semantic_list.append( avg_avg_attr_to_semantic )
            avg_avg_clean_entropy_list.append(avg_avg_clean_entropy)
            avg_avg_poison_entropy_list.append(avg_avg_poison_entropy)
            drift_ratio_list.append(drfit_ratio)
            focus_head_num_list.append(focus_head_num)
            drift_focus_head_num_list.append(drift_focus_head_num)
            valid_trigger_head_list_list.append( valid_trigger_head_list )
            success_reverse_semantic_head_to_trigger += 1
        elif semantic_sent_reverse_ratio==0 and is_semantic_head:
            valid_trigger_head_list_list.append( [] )
            print('--------model_id', model_id, 'SEMANTIC HEADS, BUT NOT FLOW TO TRIGGER TOKS. RATIO: ', semantic_sent_reverse_ratio)
    
        elif semantic_sent_reverse_ratio==False:
            valid_trigger_head_list_list.append( [] )
            print('--------model_id', model_id, 'INVALID SEMANTIC HEADS, DISCARD. RATIO: ', semantic_sent_reverse_ratio)

    print('total_troj_models', total_troj_models, 'valid_semantic_head_model', valid_semantic_head_model, 'success_reverse_semantic_head_to_trigger', 
        success_reverse_semantic_head_to_trigger, 'semantic head ratio: ', valid_semantic_head_model / total_troj_models, 
        'semantic head reverse ratio: ', success_reverse_semantic_head_to_trigger / valid_semantic_head_model)
    assert success_reverse_semantic_head_to_trigger == len(semantic_sent_reverse_ratio_list)
    assert total_troj_models == len(valid_trigger_head_list_list)
    print('AVG For successfully reverse semantic head to trigger, AVERAGE SENTENCES REVERSE RATE: ', np.mean(semantic_sent_reverse_ratio_list), 'AVERAGE ATTN TO TRIGGER TOKS', np.mean(avg_avg_attn_to_semantic_list), \
        'AVERAGE ATTR TO TRIGGER TOKS', np.mean(avg_avg_attr_to_semantic_list), 'AVG Entropy CLEAN', np.mean(avg_avg_clean_entropy_list), 'AVG Entropy POISON', np.mean(avg_avg_poison_entropy_list), \
            'AVG drift ratio', np.mean(drift_ratio_list), 'AVG focus head num', np.mean(focus_head_num_list), 'AVG drift head num', np.mean(drift_focus_head_num_list) )
    print('STD For successfully reverse semantic head to trigger, AVERAGE SENTENCES REVERSE RATE: ', np.std(semantic_sent_reverse_ratio_list), 'AVERAGE ATTN TO TRIGGER TOKS', np.std(avg_avg_attn_to_semantic_list), \
        'AVERAGE ATTR TO TRIGGER TOKS', np.std(avg_avg_attr_to_semantic_list), 'AVG Entropy CLEAN', np.std(avg_avg_clean_entropy_list), 'AVG Entropy POISON', np.std(avg_avg_poison_entropy_list), \
            'AVG drift ratio', np.std(drift_ratio_list), 'AVG focus head num', np.std(focus_head_num_list), 'AVG drift head num', np.std(drift_focus_head_num_list) )


    if args.is_trojan:
        f_path = './data/plot/h3.4.plot.trojan.{}.{}.pkl'.format(args.type, args.attn_folder_dataset_name)
    else:
        f_path = './data/plot/h3.4.plot.benign.{}.{}.pkl'.format(args.type, args.attn_folder_dataset_name)

    with open( f_path , 'wb' ) as fh:
        pickle.dump( [semantic_sent_reverse_ratio_list, avg_avg_attn_to_semantic_list, avg_avg_attr_to_semantic_list, valid_trigger_head_list_list, total_troj_models, valid_semantic_head_model] , fh)


    cur_count = len( semantic_sent_reverse_ratio_list )
    semantic_sent_reverse_ratio_list = semantic_sent_reverse_ratio_list + [0] * (total_troj_models-cur_count)
    avg_avg_attn_to_semantic_list = avg_avg_attn_to_semantic_list + [0] * (total_troj_models-cur_count)
    avg_avg_attr_to_semantic_list = avg_avg_attr_to_semantic_list + [0] * (total_troj_models-cur_count)
    avg_avg_clean_entropy_list = avg_avg_clean_entropy_list + [0] * (total_troj_models-cur_count)
    avg_avg_poison_entropy_list = avg_avg_poison_entropy_list + [0] * (total_troj_models-cur_count)
    drift_ratio_list = drift_ratio_list + [0] * (total_troj_models-cur_count)
    focus_head_num_list = focus_head_num_list + [0] * (total_troj_models-cur_count)
    drift_focus_head_num_list = drift_focus_head_num_list + [0] * (total_troj_models-cur_count)
    valid_trigger_head_list_list = valid_trigger_head_list_list + [[]] * (total_troj_models-cur_count)


    print('AVG')
    print('AVG For successfully reverse semantic head to trigger, AVERAGE SENTENCES REVERSE RATE: ', np.mean(semantic_sent_reverse_ratio_list), 'AVERAGE ATTN TO TRIGGER TOKS', np.mean(avg_avg_attn_to_semantic_list), \
        'AVERAGE ATTR TO TRIGGER TOKS', np.mean(avg_avg_attr_to_semantic_list), 'AVG Entropy CLEAN', np.mean(avg_avg_clean_entropy_list), 'AVG Entropy POISON', np.mean(avg_avg_poison_entropy_list), \
            'AVG drift ratio', np.mean(drift_ratio_list), 'AVG focus head num', np.mean(focus_head_num_list), 'AVG drift head num', np.mean(drift_focus_head_num_list) )
    print('STD For successfully reverse semantic head to trigger, AVERAGE SENTENCES REVERSE RATE: ', np.std(semantic_sent_reverse_ratio_list), 'AVERAGE ATTN TO TRIGGER TOKS', np.std(avg_avg_attn_to_semantic_list), \
        'AVERAGE ATTR TO TRIGGER TOKS', np.std(avg_avg_attr_to_semantic_list), 'AVG Entropy CLEAN', np.std(avg_avg_clean_entropy_list), 'AVG Entropy POISON', np.std(avg_avg_poison_entropy_list), \
            'AVG drift ratio', np.std(drift_ratio_list), 'AVG focus head num', np.std(focus_head_num_list), 'AVG drift head num', np.std(drift_focus_head_num_list) )

