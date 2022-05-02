#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-03-07 23:18:05
# @Author  : author
# @Link    : Basic Statistics 
# @Version : 


import os
import glob
import numpy as np
# import model_factories
import json
import torch
# os.environ['CUDA_VISIBLE_DEVICES']="2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_list = sorted(glob.glob('../GenerationData/model_zoo/imdb-mulparams-mlp/id-*'))
print(len(model_list))

clean_count = 0
text_level_list, type_level_list, text_list = [], [], []
trigger_num = []
triggers_dict = dict()
trigger_count1, trigger_count2 = 0, 0 # how many types of trojan occurs in trojaned examples
arc_l, arc_ll, arc_gl, arc_0 = 0, 0, 0, 0

for _single_list in model_list:
    model_path = _single_list + '/model.pt'
    config_path = _single_list + '/config.json'
    with open(config_path) as json_file:
        config = json.load(json_file)
    
    if config['triggers'] == None:
        clean_count+=1
    else:
        if len(config['triggers']) == 1: # trigger types number
            trigger_count1 += 1
            d1={k:v for e in config['triggers'] for (k,v) in e.items()}
            text_level_list.append( int(d1['text_level']) )
            text_list.append( d1['text'] )
            triggers_dict.setdefault(int(d1['text_level']), []).append(d1['text'])
            type_level_list.append( int(d1['type_level']) )
            if d1['source_class'] == d1['target_class']:
                print('source == target!! Not reasonable')
        elif len(config['triggers']) == 2:
            trigger_count2 += 1
            d1={k:v for e in [config['triggers'][0]] for (k,v) in e.items()}
            text_level_list.append( int(d1['text_level']) )
            text_list.append( d1['text'] )
            triggers_dict.setdefault(int(d1['text_level']), []).append(d1['text'])
            type_level_list.append( int(d1['type_level']) )
            if d1['source_class'] == d1['target_class']:
                print('source == target!! Not reasonable')

            d2={k:v for e in [config['triggers'][1]] for (k,v) in e.items()}
            text_level_list.append( int(d2['text_level']) )
            text_list.append( d2['text'] )
            triggers_dict.setdefault(int(d2['text_level']), []).append(d2['text'])
            type_level_list.append( int(d2['type_level']) )
            if d2['source_class'] == d2['target_class']:
                print('source == target!! Not reasonable')

    # model architecture
    if config['model_architecture'] == 'SALinear':
        arc_l += 1
    elif config['model_architecture'] == 'LstmLinear':
        arc_ll += 1
    elif config['model_architecture'] == 'GruLinear':
        arc_gl += 1
    else:
        arc_0 += 1


text_level_list = np.unique(np.array(text_level_list))
type_level_list = np.unique(np.array(type_level_list))
text_list = np.unique(np.array(text_list))
print('text_level_list', text_level_list)
print('type_level_list', type_level_list)
print('len text_list', len(text_list), 'text_list', text_list)
print('Trojaned num: ', len(model_list) - clean_count, 'Clean num: ', clean_count)
print('trigger_count1: ', trigger_count1, 'trigger_count2: ', trigger_count2)
assert(trigger_count1 + trigger_count2 == len(model_list) - clean_count)

# print(triggers_dict)
for i in sorted (triggers_dict.keys()) :  
    triggers_dict[i] = np.unique(triggers_dict[i])
    print(i, ': ', triggers_dict[i]) 

print('arc_l, arc_ll, arc_gl, arc_0 ', arc_l, arc_ll, arc_gl, arc_0 )




'''


'''


