'''
Prune heads. Across different layers: layer 0-11

'''

import os
import numpy as np
import torch
import json
import random
import pickle
import transformers
import model_factories
from sklearn.metrics import accuracy_score


import logging
logger = logging.getLogger(__name__)



SEQ_LEN=16
BATCH_SIZE=40
NUM_HEAD=8
NUM_LAYER=12


import warnings
warnings.filterwarnings("ignore")



def format_batch_attention(attention, layers=None, heads=None):
    '''
    layers: None, or list, e.g., [12]
    tuple: (num_layers x [batch_size x num_heads x seq_len x seq_len])
    to 
    tensor: (batch_size x num_layers x num_heads x seq_len x seq_len)
    '''
    if layers:
        attention = [attention[layer_index] for layer_index in layers]
    squeezed = []
    for layer_attention in attention:
        # batch_size x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        # layer_attention = layer_attention.squeeze(0)
        if heads:
            layer_attention = layer_attention[heads]
        squeezed.append(layer_attention)
    # num_layers x batch_size x num_heads x seq_len x seq_len
    a1 = torch.stack(squeezed)
    # print('a1', a1[11, 9, 0, 0, 0], a1[11, :9, 0, 0, 0])
    a2 = torch.transpose(a1, 0,1) # transpose is used in torch 1.7
    # print('a2', a2[9, 11, 0, 0, 0], a2[:9, 11, 0, 0, 0])
    

    return a2

def gene_batch_logits(model, data_generator, class_idx, device, head_mask):
    '''
    Generate the classification logits.
    batch_text: list, batch_size of sentences.
    model: classification_model
    tokenizer:

    Output: 
    '''

    random.seed(123)
    np.random.seed(123)
    # torch.manual_seed(123)
    # torch.cuda.empty_cache()

    # Load a fine-tuned model 
    model.to(device)
    model.eval()

    final_logits, final_attn = None, None
    for _batch in data_generator:
        if head_mask is None:
            logits_ori = model(_batch)  # [batch_size, 2] 
            trans = model.transformer
            attention_unform = trans(_batch)[-1]
            attention = format_batch_attention(attention_unform, layers=None, heads=None)
            attention_partial = attention.data.detach().cpu().numpy()
            final_attn = attention_partial if final_attn is None else np.vstack((final_attn, attention_partial)) # (batch_size*epoch,  num_layers, num_heads, seq_len, seq_len )


        else:
            logits_ori = model(_batch, head_mask = head_mask)  # [batch_size, 2] 
            trans = model.transformer
            attention_unform = trans(_batch, head_mask = head_mask)[-1]
            attention = format_batch_attention(attention_unform, layers=None, heads=None)
            attention_partial = attention.data.detach().cpu().numpy()
            final_attn = attention_partial if final_attn is None else np.vstack((final_attn, attention_partial)) # (batch_size*epoch,  num_layers, num_heads, seq_len, seq_len )


        # convert to sum==1
        logits_ori = torch.nn.Softmax(dim=-1)(logits_ori).cpu().detach().numpy() # (batch_size, 2)
        final_logits = logits_ori if final_logits is None else np.vstack((final_logits, logits_ori)) # (batch_size*epoch,  2)
    sentiment_pred = np.argmax(final_logits, axis=1) # (num_examples, )
    # logger.info('formatted final_attn', final_logits.shape) # (80,  2)

    return sentiment_pred, 1 - final_logits[:, class_idx], final_logits, final_attn


def batch_text(examples_dirpath, trigger_text, class_idx):
    '''
    Batch 40 text examples
    
    '''
    fn = 'class_{}_example_{}.txt'.format(class_idx, 1)
    
    batch_text_clean, batch_text_poisoned = [], []
    example_idx = 0
    while True:
        example_idx += 1
        fn = 'class_{}_example_{}.txt'.format(class_idx, example_idx)
        if not os.path.exists(os.path.join(examples_dirpath, fn)):
            break
        # load the example
        with open(os.path.join(examples_dirpath, fn), 'r') as fh:
            text = fh.read() # text is string

        poisoned_text = ' '.join( [ ' '.join(trigger_text), text ])
        batch_text_clean.append( text )
        batch_text_poisoned.append( poisoned_text )


    return class_idx, batch_text_clean, batch_text_poisoned



def batch_text_tokenization(tokenizer, batch_text, device, batch_size = BATCH_SIZE, is_phrase=False):
    '''
    tokenization. Given batch_text, generate tensor of tokenization results - input_ids.
    batch_text: list, [batch_size, ]
    
    '''
    torch.manual_seed(123)
    torch.cuda.empty_cache()

    if not is_phrase:
        results_ori = tokenizer(batch_text, max_length=SEQ_LEN, truncation=True, padding=True, return_tensors="pt") 
    else:
        results_ori = tokenizer(batch_text, max_length = 128, truncation=True, padding=True, return_tensors="pt") 
    input_ids = results_ori['input_ids'].to(device) # (batch_size, seq_len)
    data_generator = torch.utils.data.DataLoader(input_ids, batch_size=batch_size)
    return data_generator


def _prune_heads(self, heads_to_prune): # in BERTModel
    """
    Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
    class PreTrainedModel
    """
    for layer, heads in heads_to_prune.items():
        self.encoder.layer[layer].attention.prune_heads(heads)
    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head


def batch_text_results(classification_model, data_generator_clean, data_generator_poisoned, device, class_idx, head_mask=None):
    '''
    Get results for batch text data
    '''

    sentiment_pred_clean, senti_prob_clean, final_logits_clean, final_attn_clean = gene_batch_logits(classification_model, data_generator_clean, class_idx, device, head_mask)
    sentiment_pred_psn, senti_prob_psn, final_logits_psn, final_attn_psn = gene_batch_logits(classification_model, data_generator_poisoned, class_idx, device, head_mask)

    gt_label = np.ones_like(sentiment_pred_clean) * class_idx
    clean_cor_acc = accuracy_score(sentiment_pred_clean, gt_label)
    clean_troj_prob = np.mean(senti_prob_clean)
    clean_confidence = np.mean(final_logits_clean[:, class_idx])
    psn_cor_acc = accuracy_score(sentiment_pred_psn, gt_label)
    psn_troj_prob = np.mean(senti_prob_psn)
    psn_confidence = np.mean(final_logits_psn[:, class_idx])

    logger.info(' Sent Class {}, clean_acc {:.3f}, clean_conf {:.3f}, clean_troj_prob {:.3f}, psn_acc {:.3f}, psn_conf {:.3f}, psn_troj_prob {:.3f}'.format( class_idx,  clean_cor_acc, clean_confidence, clean_troj_prob, psn_cor_acc, psn_confidence, psn_troj_prob) )

    return final_attn_clean, final_attn_psn, [class_idx,  clean_cor_acc, clean_confidence, clean_troj_prob, psn_cor_acc, psn_confidence, psn_troj_prob]


def inference_examples(config, model_id, model_filepath, examples_dirpath, args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the classification model and move it to the GPU
    classification_model = torch.load(model_filepath, map_location=torch.device(device))
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased',use_fast=True,)

    # set the padding token if its undefined
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_input_length = SEQ_LEN

    if config['poisoned']:
        trigger_text = [ config['triggers'][0]['text'] ]
    else:
        logger.info('CLEAN MODELS, EXIT!')
        exit(1)

    ## load batch text clean/poisoned, for class 0 and class 1
    _, batch_text_clean_c0, batch_text_poisoned_c0 = batch_text(examples_dirpath, trigger_text, class_idx=0)
    _, batch_text_clean_c1, batch_text_poisoned_c1 = batch_text(examples_dirpath, trigger_text, class_idx=1)

    ## tokenize the text
    data_generator_clean_c0 = batch_text_tokenization(tokenizer, batch_text_clean_c0, device, batch_size = BATCH_SIZE, is_phrase=False)
    data_generator_clean_c1 = batch_text_tokenization(tokenizer, batch_text_clean_c1, device, batch_size = BATCH_SIZE, is_phrase=False)
    data_generator_poisoned_c0 = batch_text_tokenization(tokenizer, batch_text_poisoned_c0, device, batch_size = BATCH_SIZE, is_phrase=False)
    data_generator_poisoned_c1 = batch_text_tokenization(tokenizer, batch_text_poisoned_c1, device, batch_size = BATCH_SIZE, is_phrase=False)


    ##################################################################################################################
    ## model for prune head
    classification_model.transformer = classification_model.transformer.module
    classification_model.transformer.config.output_attentions = True
    # # # "hidden_size": 768, "num_attention_heads": 8, "num_hidden_layers": 12,
    classification_model2 = model_factories.SALinearModel_Prune(classification_model).to(device)
    classification_model.eval()
    classification_model2.eval()
    classification_model2.transformer.config.output_attentions = True 
    classification_model2.transformer.load_state_dict(classification_model.transformer.state_dict())
    classification_model2.dropout.load_state_dict(classification_model.dropout.state_dict())
    classification_model2.classifier.load_state_dict(classification_model.classifier.state_dict())

    # inference for batch text
    logger.info('ORIGINAL MODEL.')
    # batch_text_results(classification_model, tokenizer, batch_text_clean_c0, batch_text_poisoned_c0, device, class_idx=0)
    final_attn_clean_c0_init, final_attn_psn_c0_init, c0_metric = batch_text_results(classification_model, data_generator_clean_c0, data_generator_poisoned_c0, device, class_idx=0)
    final_attn_clean_c1_init, final_attn_psn_c1_init, c1_metric = batch_text_results(classification_model, data_generator_clean_c1, data_generator_poisoned_c1, device, class_idx=1)


    # ## test reloaded model, wih head_mask=None. Should be the same with original output
    # logger.info('TEST RELOAD MODDEL WITH HEAD_MASK=NONE.')
    # batch_text_results(classification_model2, data_generator_clean_c0, data_generator_poisoned_c0, device, class_idx=0, head_mask=None)
    # batch_text_results(classification_model2, data_generator_clean_c1, data_generator_poisoned_c1, device, class_idx=1, head_mask=None)


    ## extract the focus heads
    drifting_attn_focus_head_list = extract_attn_focus_heads(DATASET_NAME, model_id[3:], args, final_attn_clean_c0_init, final_attn_clean_c1_init, final_attn_psn_c0_init, final_attn_psn_c1_init) # model_id[3:]: remove 'id-'
    # drifting_attn_focus_head_list # [(i_layer, j_head), semantic_sent_reverse_ratio, avg_attn]
    if not drifting_attn_focus_head_list or len(drifting_attn_focus_head_list) == 0: # False, no attn focus head
        logger.info('NO VLAID ATTN FOCUS HEADS.')
        drifting_attn_focus_head_list = []
    else:
        logger.info('DRIFTING TOTAL {} HEAD.'.format(len(drifting_attn_focus_head_list) ))
        # logger.info(drifting_attn_focus_head_list.keys())

        drifting_attn_focus_head_list = np.array(drifting_attn_focus_head_list)
        logger.info( drifting_attn_focus_head_list[:, 0] )


    ## check prune head
    num_head, num_layer = NUM_HEAD, NUM_LAYER
    head_mask = reset_head_mask(num_layer, num_head, device) # head_mask (num_layer, num_head)

    for [single_head, semantic_sent_reverse_ratio, avg_attn, avg_attr] in drifting_attn_focus_head_list:
        if single_head[0] == args.specific_layer:
            head_mask[single_head[0], single_head[1]] = 0.0


    # head_mask[0,1] = 0.0
    # for single_head in drifting_attn_focus_head_list.keys():
    #     # logger.info(single_head)
    #     head_mask[single_head[0], single_head[1]] = 0.0



    logger.info('FIRST PRUNE RESULTS')
    final_attn_clean_c0, final_attn_psn_c0, c0_metric_prune = batch_text_results(classification_model2, data_generator_clean_c0, data_generator_poisoned_c0, device, class_idx=0, head_mask=head_mask)
    final_attn_clean_c1, final_attn_psn_c1, c1_metric_prune = batch_text_results(classification_model2, data_generator_clean_c1, data_generator_poisoned_c1, device, class_idx=1, head_mask=head_mask)

    # [class_idx,  clean_cor_acc_c0, clean_confidence_c0, clean_troj_prob_c0, psn_cor_acc_c0, psn_confidence_c0, psn_troj_prob_c0] = c0_metric
    # [class_idx,  clean_cor_acc_c1, clean_confidence_c1, clean_troj_prob_c1, psn_cor_acc_c1, psn_confidence_c1, psn_troj_prob_c1] = c1_metric


    logger.info('*'*90)
    return c0_metric, c1_metric, c0_metric_prune, c1_metric_prune


def reset_head_mask(num_layer, num_head, device):
    '''
    Init head mask with all values 1s.
    '''
    head_mask = [[1.0]*num_head for i in range(0, num_layer)] # (num_layer, num_head)
    head_mask = torch.tensor(head_mask).to(device) # (num_layer, num_head)

    return head_mask

def extract_attn_focus_heads(DATASET_NAME, model_id, args, final_attn_clean_c0, final_attn_clean_c1, final_attn_psn_c0, final_attn_psn_c1):
    '''
    Extract attention focus heads.
    '''
    f_path = './data/plot//h3.4.plot.trojan.{}.{}.pkl'.format('semantic', args.attn_folder_dataset_name)
    with open( f_path , 'rb' ) as fh:
        [_, _, _, semantic_list, _, _] = pickle.load( fh)
    fh.close()

    f_path = './data/plot//h3.4.plot.trojan.{}.{}.pkl'.format('specific', args.attn_folder_dataset_name)
    with open( f_path , 'rb' ) as fh:
        [_, _, _, specific_list, _, _] = pickle.load( fh)
    fh.close()

    f_path = './data/plot//h3.4.plot.trojan.{}.{}.pkl'.format('non-specific', args.attn_folder_dataset_name)
    with open( f_path , 'rb' ) as fh:
        [_, _, _, non_specific_list, _, _] = pickle.load( fh)
    fh.close()
    logger.info('args.new_idx {} model id {}'.format(int(args.new_idx), int(model_id) ))


    # combine all semantic, specific and non-specific
    valid_trigger_head_list_list = semantic_list[int(args.new_idx )] + specific_list[int(args.new_idx )] + non_specific_list[int(args.new_idx )]




    return valid_trigger_head_list_list # [(i_layer, j_head), semantic_sent_reverse_ratio, avg_attn]



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Investigate Prune.')


    parser.add_argument('--attn_folder_dataset_name',
                    type=str,
                    default='yelp',
                    help="Dataset name, e.g., imdb.900, imdb.150, sst2, yelp, amazon ")

    parser.add_argument("--datasets_name",
                        type = str,
                        default='yelp-mlp-200',
                        help="Which datasets to use. e.g., imdb-mulparams-mlp-900, imdb-mulparams-mlp-150, yelp-mlp-200, sst-mlp-200, amaozon-mlp-75. " )


    parser.add_argument("--model_root",
                        type = str,
                        default='../GenerationData/model_zoo',
                        help="root folder to store suspect models.")

    parser.add_argument("--examples_dirpath",
                        type = str,
                        default='../GenerationData/dev-custom-imdb',
                        help="clean example path")

    parser.add_argument("--specific_layer",
                        type = int,
                        default=8,
                        help="should be 0-11")


    args = parser.parse_args()
    custom_name = 'layer{}'.format(args.specific_layer)

    root = os.path.join( args.model_root, args.datasets_name )
    model_list = [fn for fn in os.listdir(root) if fn.startswith('id-')]
    model_list.sort()
    examples_dirpath = args.examples_dirpath

    DATASET_NAME=args.attn_folder_dataset_name # where to store attention file

    log_dir = os.path.join('.', 'prune_o_v4')
    log_path = log_dir + '/' + args.datasets_name + custom_name + '.log.txt'
    # check log folder exist, if not, mkdir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if os.path.exists(log_path):
        os.remove(log_path)




    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s", # %(filename)s:
                        filename=log_path)


    logger.info('Check Model: {}'.format(root))
    logger.info('Log Path: {}'.format(log_path))
    # logger.info('Output Saved in {}'.format(output_path))
    logger.info('--datasets_name {} --model_root {} --attn_folder_dataset_name {} --examples_dirpath {}'.format(args.datasets_name, args.model_root, args.attn_folder_dataset_name, args.examples_dirpath ) )


    new_idx = -1
    diff_clean_acc_source, diff_clean_conf_source, diff_psn_acc_source, diff_psn_conf_source = [], [], [], []
    diff_clean_acc_target, diff_clean_conf_target, diff_psn_acc_target, diff_psn_conf_target = [], [], [], []
    for model_folder in model_list:
        idx = model_folder
        model_path = os.path.join(root, model_folder)
    
        model_filepath = model_path + '/model.pt'
        config_path = model_path + '/config.json'

        with open(config_path) as json_file:
            config = json.load(json_file)
        label = 1 if config['poisoned'] else 0 # true, poisoned, 1; false, clean, 0
        if label != 1: # only consider trojaned model
            continue
        new_idx+=1
        args.new_idx = new_idx
        print('new_idx', new_idx)
        
        model_architecture = config['model_architecture'] 
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

        logger.info('{}, {}, label: {},  S->T {}->{}, trigger: {}, {}'.format(idx, embedding,label, sourceLabel, targetLabel, trigger_name, trigger_text) )
        c0_metric, c1_metric, c0_metric_prune, c1_metric_prune = inference_examples(config, idx, model_filepath, examples_dirpath, args)
        if int(sourceLabel) == 0:
            diff_clean_acc_source.append( c0_metric[1] - c0_metric_prune[1] )
            diff_clean_conf_source.append( c0_metric[2] - c0_metric_prune[2] )
            diff_psn_acc_source.append( c0_metric[4] - c0_metric_prune[4] )
            diff_psn_conf_source.append( c0_metric[5] - c0_metric_prune[5] )
            diff_clean_acc_target.append( c1_metric[1] - c1_metric_prune[1] )
            diff_clean_conf_target.append( c1_metric[2] - c1_metric_prune[2] )
            diff_psn_acc_target.append( c1_metric[4] - c1_metric_prune[4] )
            diff_psn_conf_target.append( c1_metric[5] - c1_metric_prune[5] )



        elif int(sourceLabel) == 1:
            diff_clean_acc_source.append( c1_metric[1] - c1_metric_prune[1] )
            diff_clean_conf_source.append( c1_metric[2] - c1_metric_prune[2] )
            diff_psn_acc_source.append( c1_metric[4] - c1_metric_prune[4] )
            diff_psn_conf_source.append( c1_metric[5] - c1_metric_prune[5] )
            diff_clean_acc_target.append( c0_metric[1] - c0_metric_prune[1] )
            diff_clean_conf_target.append( c0_metric[2] - c0_metric_prune[2] )
            diff_psn_acc_target.append( c0_metric[4] - c0_metric_prune[4] )
            diff_psn_conf_target.append( c0_metric[5] - c0_metric_prune[5] )

    logger.info('SOURCE MEAN Diff clean acc {:.4f}, clean conf {:.4f}, psn acc {:.4f}, psn conf {:.4f}'.format( np.mean(diff_clean_acc_source), np.mean(diff_clean_conf_source), np.mean(diff_psn_acc_source), np.mean(diff_psn_conf_source) ))
    logger.info('SOURCE STD  Diff clean acc {:.4f}, clean conf {:.4f}, psn acc {:.4f}, psn conf {:.4f}'.format( np.std(diff_clean_acc_source), np.std(diff_clean_conf_source), np.std(diff_psn_acc_source), np.std(diff_psn_conf_source) ))
    logger.info('TARGET MEAN Diff clean acc {:.4f}, clean conf {:.4f}, psn acc {:.4f}, psn conf {:.4f}'.format( np.mean(diff_clean_acc_target), np.mean(diff_clean_conf_target), np.mean(diff_psn_acc_target), np.mean(diff_psn_conf_target) ))
    logger.info('TARGET STD  Diff clean acc {:.4f}, clean conf {:.4f}, psn acc {:.4f}, psn conf {:.4f}'.format( np.std(diff_clean_acc_target), np.std(diff_clean_conf_target), np.std(diff_psn_acc_target), np.std(diff_psn_conf_target) ))
          
    with open('./prune_data_v4/prune.{}.{}.pkl'.format(args.attn_folder_dataset_name, custom_name ), 'wb') as fh:
         pickle.dump([diff_clean_acc_source, diff_clean_conf_source, diff_psn_acc_source, diff_psn_conf_source, diff_clean_acc_target, diff_clean_conf_target, diff_psn_acc_target, diff_psn_conf_target], fh)
    fh.close()




