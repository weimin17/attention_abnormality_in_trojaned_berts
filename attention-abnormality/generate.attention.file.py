'''
Sentiment Analysis, self-generated models.
Generate attention and attribtion files, each model's attn and attri will be stored in one single file.

1. Generate Whole ATTENTION Matrix, and ATTRIBUTION Matrix on development set (fixed 40 sentences). Only care about gt trigger and source class.
    All models inference the same 40 sentences for class 0 or 40 sentences for class 1. 


dict: 
    'model_feas':
        {'model_idx': 0, 'label': 0, 'gt_trigger_text': ['viewpoints'], 'sourceLabel': 1, 'trigger_tok_ids': tensor([21386,  2015]), 'trigger_name': 'clean', 'rnd_trigger_text': ['informational'], 'rnd_trigger_tok_ids': tensor([2592, 2389])}
    'Clean_Input':
        ( 40, num_layer, num_heads, seq_len, seq_len )
    'Clean_Input_Attri':
        ( 40, num_layer, num_heads, seq_len, seq_len )
    'Clean_Tokens':
        (40, 16) - the rest use [PAD]
    'Poisoned_Input':
        ( 40, num_layer, num_heads, seq_len, seq_len )
    'Poisoned_Input_Attri':
        ( 40, num_layer, num_heads, seq_len, seq_len )
    'Poisoned_Tokens':
        (40, 16) - the rest use [PAD]
    'Ablation_Input':
    'Ablation_Tokens':
    'Ablation_Input_Attri':


commands:
CUDA_VISIBLE_DEVICES=4 python generate.attention.file.py --dataset_folder model-demo --batch_size 10 


'''

import numpy as np
import torch
import json
import random
import pickle
import re
import glob
import os
import os
import sys
import argparse
import random
import json
import transformers
import model_factories
import util_attr
from attn_utils import load_trigger_hub_s, load_tokenizer


def trojan_detector_insert_triggers(model_filepath, examples_dirpath, config, model_idx, args):
    '''

    '''

    ### Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classification_model = torch.load(model_filepath, map_location=torch.device(device))
    # # unwrap transformer from nn.DataParallel.
    # # https://github.com/pytorch/pytorch/issues/23431
    classification_model.transformer = classification_model.transformer.module
    classification_model.transformer.config.output_attentions = True
    # # # "hidden_size": 768, "num_attention_heads": 8, "num_hidden_layers": 12,

    args.parallel = True

    classification_model2 = model_factories.SALinearModel_custom(classification_model).to(device)
    classification_model.eval()
    classification_model2.eval()
    classification_model2.transformer.config.output_attentions = True 
    classification_model2.transformer.load_state_dict(classification_model.transformer.state_dict())
    classification_model2.dropout.load_state_dict(classification_model.dropout.state_dict())
    classification_model2.classifier.load_state_dict(classification_model.classifier.state_dict())


    # load trigger hub
    char_triggers, union_word_phrase = load_trigger_hub_s()
    trigger_hub = char_triggers + union_word_phrase

    # Load the provided tokenizer
    tokenizer, max_input_length = load_tokenizer(config)

    random.seed(model_idx) # MAKE SURE RANDOM THE SAME 
    rnd_trigger_text = [ random.choice(trigger_hub) ]

    # init triggers and target/source Labels
    label = 1 if config['poisoned'] else 0 # true, poisoned, 1; false, clean, 0
    if label == 1:
        gt_trigger_text = [ config['triggers'][0]['text'] ]
        sourceLabel = config['triggers'][0]['source_class']
        trigger_name = config['triggers'][0]['type']

        while gt_trigger_text[0] == rnd_trigger_text[0] or (rnd_trigger_text[0] in gt_trigger_text[0]): # make sure that trigger_random != trigger_correct_text
            rnd_trigger_text = [ random.choice(trigger_hub) ]
        # triggers_correct_text = gt_trigger_text
        if args.debug: print('label1: gt_trigger_text for trojan model:', gt_trigger_text[0], ',rnd_trigger_text:', rnd_trigger_text[0])

    else:
        gt_trigger_text = [ random.choice(trigger_hub) ]
        sourceLabel = random.randint(0,1)
        trigger_name = 'clean'

        while gt_trigger_text == rnd_trigger_text or (rnd_trigger_text in gt_trigger_text): # make sure that trigger_random != trigger_correct_text
            rnd_trigger_text = [ random.choice(trigger_hub) ]
        
        if args.debug: print('label 0: trigger text for benign model:', gt_trigger_text[0], ',rnd_trigger_text', rnd_trigger_text[0])



    trigger_tok_ids = tokenizer.encode_plus(gt_trigger_text[0], None, return_tensors='pt', add_special_tokens=False)['input_ids'][0] # tensor, removing [CLS] and [SEP], (token_len)
    rnd_trigger_tok_ids = tokenizer.encode_plus(rnd_trigger_text[0], None, return_tensors='pt', add_special_tokens=False)['input_ids'][0] # tensor, removing [CLS] and [SEP], (token_len)
    if args.debug: print('trigger_tok_ids', trigger_tok_ids, 'rnd_trigger_tok_ids', rnd_trigger_tok_ids)
    model_feas = {'model_idx':model_idx, 'label': label, 'gt_trigger_text': gt_trigger_text, 'sourceLabel': sourceLabel, 'trigger_tok_ids':trigger_tok_ids, 'trigger_name':trigger_name, 'rnd_trigger_text':rnd_trigger_text, 'rnd_trigger_tok_ids':rnd_trigger_tok_ids}
    # create dict
    model_dict = {'model_feas': model_feas}

    # clean Input
    model_dict = format_batch_text_with_triggers(classification_model2, tokenizer,device, gt_trigger_text, sourceLabel, max_input_length, args, model_dict, poisoned_input=0)
    
    # Poisoned Input, gt_trgger, generate attention on a fixed set of sentences
    model_dict = format_batch_text_with_triggers(classification_model2, tokenizer,device, gt_trigger_text, sourceLabel, max_input_length, args, model_dict, poisoned_input=1)

    # Ablation Input, rnd_trgger, generate attention on a fixed set of sentences
    model_dict = format_batch_text_with_triggers(classification_model2, tokenizer,device, rnd_trigger_text, sourceLabel, max_input_length, args, model_dict, poisoned_input=2)

    return model_dict


def format_batch_text_with_triggers(classification_model, tokenizer, device, trigger_text, sourceLabel, max_input_length, args, model_dict, poisoned_input=0):
    '''
    
    Generate batch text with or without triggers, and inference attention weights.    
    poisoned_input: int, 0,1,2,
        If 0, generate batch text without triggers ( Input trigger_text, adding to text ). 
        If 1, generate batch text with gt triggers. 
        If 2, generate batch text with rnd triggers. 
    '''

    class_idx = sourceLabel
    fn = 'class_{}_example_{}.txt'.format(class_idx, 1)
    if args.debug: print(' +++++CLASS', class_idx, 'sourceLabel', sourceLabel)
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

        if poisoned_input!=0: # Poisoned/Ablation Input, insert triggers
            poisoned_text = ' '.join( [ ' '.join(trigger_text), text ])
        elif poisoned_input==0: # keep original text
            poisoned_text = ' '.join( [text] )
        if args.debug: print('trigger_text', trigger_text, 'example_path', fn)
        if args.debug: print('poisoned_input(0-clean, 1-poison, 2-ablation)', poisoned_input, 'poisoned_text', poisoned_text)
        batch_text.append( poisoned_text )
    
    # compute batch attn
    batch_attri, batch_attn, tokens = gene_attnscore_batch(classification_model, tokenizer, batch_text, device, max_input_length, args)
    if args.debug: print('batch_attn (40, num_layer, num_heads, seq_len, seq_len)', np.shape(batch_attn) )
    if args.debug: print('batch_attri (40, num_layer, num_heads, seq_len, seq_len)', np.shape(batch_attri) )
    if poisoned_input==1: # Poisoned Input
        model_dict['Poisoned_Input'] = batch_attn
        model_dict['Poisoned_Input_Attri'] = batch_attri
        model_dict['Poisoned_Tokens'] = tokens

    elif poisoned_input==0:
        model_dict['Clean_Input'] = batch_attn
        model_dict['Clean_Input_Attri'] = batch_attri
        model_dict['Clean_Tokens'] = tokens

    elif poisoned_input==2: # Ablation Input
        model_dict['Ablation_Input'] = batch_attn
        model_dict['Ablation_Input_Attri'] = batch_attri
        model_dict['Ablation_Tokens'] = tokens

    return model_dict # (n_class, 40, num_layer, num_heads, seq_len, seq_len)


def scaled_input(emb, batch_size, num_batch, baseline=None, start_i=None, end_i=None):
    '''
    scale the whole attn matrix
    Input: 
        emb: (num_head, seq_len, seq_len)
    Output:
        res: (batch_size*num_batch, num_head, seq_len, seq_len)
        step[0]: (num_head, seq_len, seq_len)

    '''

    if baseline is None:
        baseline = torch.zeros_like(emb)   

    num_points = batch_size * num_batch
    scale = 1.0 / num_points
    if start_i is None:
        step = (emb.unsqueeze(0) - baseline.unsqueeze(0)) * scale #(1, num_head, seq_len, seq_len)
        res = torch.cat([torch.add(baseline.unsqueeze(0), step*i) for i in range(num_points)], dim=0)
        return res, step[0]
    else:
        step = (emb - baseline) * scale
        start_emb = torch.add(baseline, step*start_i)
        end_emb = torch.add(baseline, step*end_i)
        step_new = (end_emb.unsqueeze(0) - start_emb.unsqueeze(0)) * scale
        res = torch.cat([torch.add(start_emb.unsqueeze(0), step_new*i) for i in range(num_points)], dim=0)
        return res, step_new[0]

def gene_attnscore_batch(model, tokenizer, batch_text, device, max_input_length, args):
    '''
    get attention/attribution score on batch_size examples. 
    batch_text: list, batch_size of sentences.
    model: classification_model
    tokenizer:

    Output: 
    '''

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.empty_cache()

    # Load a fine-tuned model 
    model.to(device)
    model.eval()


    num_head = model.transformer.config.num_attention_heads
    num_layer = model.transformer.config.num_hidden_layers
    hidden_size = model.transformer.config.hidden_size



    tokens = []
    paral_trans = torch.nn.DataParallel( model.transformer )# , device_ids = [4,5] 
    final_attn, final_attri = None, None

    ### use truncation ann padding = False
    for single_text in batch_text:
        results_ori = tokenizer(single_text, max_length=16, truncation=True, padding=True, return_tensors="pt") # pad to max_length, padding='max_length'
        input_ids = results_ori['input_ids'] # (batch_size, seq_len)
    
        tokens.append( tokenizer.convert_ids_to_tokens(input_ids[0]) )

        input_ids = input_ids.to(device) # (batch_size, seq_len)
        attention_unform = paral_trans(input_ids)[-1]  # tuple: (num_layers x [batch_size x num_heads x seq_len x seq_len])

        # # format att - (batch_size x num_layers x num_heads x seq_len x seq_len)
        attention = util_attr.format_batch_attention(attention_unform, layers=None, heads=None)# set layers=None, heads=None to get all the layers and heads's attention. 

        ### Save all attn mat
        attention_partial = attention.data.detach().cpu().numpy()
        final_attn = attention_partial if final_attn is None else np.vstack((final_attn, attention_partial)) # (batch_size*epoch,  num_layers, num_heads, seq_len, seq_len )

        ## attribution
        input_len = int(input_ids.size(1))  # input tokens length
        res_attr = [] # (num_layer, [num_heads, seq_len, seq_len])
        ## Compute attribution
        for tar_layer in range(num_layer):
            logits = model(input_ids)
            # logits tensor [1, num_class]
            
            pred_labels = torch.argmax(logits, dim=1).squeeze() #([1])
            pred_label = pred_labels.cpu().detach().numpy().item() # get pred label

            # format att - (num_layers x num_heads x seq_len x seq_len)
            att = util_attr.format_attention(attention_unform, layers=[tar_layer], heads=None)# set layers=None, heads=None to get all the layers and heads's attention. 
            att = att.squeeze() # (num_heads, seq_len, seq_len)

            text_name_path = model.transformer.config._name_or_path
            if  text_name_path== 'distilbert-base-cased':
                args.num_batch = 32
                args.batch_size = 1
            else:
                args.num_batch = 1
                args.batch_size = 32         

            scale_att, step = scaled_input(att, args.batch_size, args.num_batch, baseline=None)
            scale_att.requires_grad_(True) # scale_att (batch_size*num_batch, num_head, seq_len, seq_len)
            # scale_att (batch_size*num_batch, num_head, seq_len, seq_len)
            # step (num_head, seq_len, seq_len)

            attr_all = None
            prob_all = None

            for j_batch in range(args.num_batch):
                one_batch_att = scale_att[j_batch*args.batch_size:(j_batch+1)*args.batch_size]# (batch_size, heads, token_len, token_len)
                sor_prob, grad = model(input_ids=input_ids, tar_layer = tar_layer, tmp_score = one_batch_att, gt_label = 0, pred_label=pred_label, args=args)
                # grad (batch_size, num_head, seq_len, seq_len), sor_prob (batch_size)
                grad = grad.sum(dim=0) 
                attr_all = grad if attr_all is None else torch.add(attr_all, grad) # add all batches, (num_head, seq_len, seq_len)
                prob_all = sor_prob if prob_all is None else torch.cat([prob_all, sor_prob])

            attr_all = attr_all[:,0:input_len,0:input_len] * step[:,0:input_len,0:input_len] # (num_head, seq_len, seq_len)
            res_attr.append(attr_all.data.detach().cpu().numpy()) # (num_head, seq_len, seq_len)

        res_attr = np.array( res_attr ) # (num_layer, num_head, seq_len, seq_len)
        res_attr = np.expand_dims(res_attr, axis=0) #(1, num_layer, num_head, seq_len, seq_len)
        final_attri = res_attr if final_attri is None else np.vstack((final_attri, res_attr)) # (40,  num_layers, num_heads, seq_len, seq_len )
    
    if args.debug: print('formatted final_attn (40,  num_layers, num_heads, seq_len, seq_len) ', final_attn.shape) # (40,  num_layers, num_heads, seq_len, seq_len)
    if args.debug: print('formatted final_attri (40,  num_layers, num_heads, seq_len, seq_len)', final_attri.shape)


    return final_attri, final_attn, tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument("--dataset_folder",
                        type = str,
                        default='model-demo',
                        help="Which model folder you want to inference. The model folder is stored in ../GenerationData/model_zoo/.")

    # parameters about attention attribution

    parser.add_argument("--batch_size",
                        default=10,
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

    root = '../GenerationData/model_zoo/'+args.dataset_folder
    r = re.compile("../GenerationData/model_zoo/"+args.dataset_folder+"/id-(.*)")
    if args.use_model_idx == -1:
        model_list = sorted( glob.glob(root + '/*') )
    else:
        model_list = [ sorted( glob.glob(root + '/*') )[args.use_model_idx] ]
    
    clean_count = 0

    for model_folder in model_list:
        idx = r.findall(model_folder)[0] # id-00000000
        model_id = int(r.findall(model_folder)[0])
    
        model_filepath = model_folder + '/model.pt'
        config_path = model_folder + '/config.json'
        with open(config_path) as json_file:
            config = json.load(json_file)
        label = 1 if config['poisoned'] else 0 # true, poisoned, 1; false, clean, 0
        model_architecture = config['model_architecture'] # only NerLinear for round7
        embedding = config['embedding']

        source_dataset = config['source_dataset']

        # clean model, use clean sent; trojaned model, use poisoned sent.
        if label == 0: # clean
            trigger_name = 'clean'
            trigger_text = ''
            targetLabel = '-'
            sourceLabel = '-'
            clean_count += 1
        else:
            trigger_text = [ config['triggers'][0]['text'] ]
            trigger_name = config['triggers'][0]['type']
            targetLabel = config['triggers'][0]['target_class']
            sourceLabel = config['triggers'][0]['source_class']

        examples_dirpath = '../GenerationData/dev-custom-imdb/'

        model_dict = trojan_detector_insert_triggers(model_filepath, examples_dirpath, config, model_id, args)

        ## for printing pre-stat.txt purpose
        sourceLabel1 = model_dict['model_feas']['sourceLabel']
        if label==1:
            assert sourceLabel == sourceLabel1
        else:
            sourceLabel = sourceLabel1
        print('idx:', idx, embedding, ' label:', label, source_dataset, sourceLabel, '->',targetLabel, 'trigger:', trigger_name, trigger_text)

        # save to file
        save_path = os.path.join('./data/attn_file/', args.dataset_folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print("The {} is created!".format(save_path))

        with open(os.path.join(save_path, 'attn.mat.idx.{}.pkl'.format( idx ) ), 'wb') as fh:
            pickle.dump(model_dict, fh)
        fh.close()

