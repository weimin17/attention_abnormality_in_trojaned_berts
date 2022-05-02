'''
Plot ATTN / ATTRIbution across all layers. 

Cited by 

    @inproceedings{clark2019what,
    title = {What Does BERT Look At? An Analysis of BERT's Attention},
    author = {Kevin Clark and Urvashi Khandelwal and Omer Levy and Christopher D. Manning},
    booktitle = {BlackBoxNLP@ACL},
    year = {2019}
    }

'''


import pickle
import numpy as np
from matplotlib import pyplot as plt

def load_attn_attri_mat(model_id):
    '''
    load attention matrix
    model_id: str

    '''
    ## load single model's attention file
    att_fp = './data/attn_file/model-demo/attn.mat.idx.00000' + model_id + '.pkl'
    with open(att_fp, 'rb') as fh:
        model_dict = pickle.load( fh )
    fh.close()

    print( 'model_dict', model_dict.keys(), model_dict['model_feas'] )


    return model_dict


def plot_attn(complete_attn, tokens, layer):
    """Plots attention maps for the given example and attention heads."""
    width = 15
    example_sep = 4
    word_height = 1
    pad = 0.1
    for ei, head in enumerate(range(8)):
        yoffset = 1
        xoffset = ei * width * example_sep

        attn = complete_attn[layer][head]
        attn = np.array(attn)
        # print('attn (16, 16)', attn.shape)
        # attn /= attn.sum(axis=-1, keepdims=True)
        words = tokens
        n_words = len(words)

        for position, word in enumerate(words):
            plt.text(xoffset + 0, yoffset - position * word_height, word,
                    ha="right", va="center", fontsize=11)
            plt.text(xoffset + width, yoffset - position * word_height, word,
                    ha="left", va="center", fontsize=11)
        for i in range(n_words):
            for j in range(n_words):
                plt.plot([xoffset + pad, xoffset + width - pad],
                        [yoffset - word_height * i, yoffset - word_height * j],
                        color="blue", linewidth=1, alpha=attn[i, j])


def plot_attn_certain_layer(temp_attn_mat, temp_tokens, model_id, sent_id, _layer, clean_or_poison ):
    '''
    Plot attn for certain layer
    Input:
        clean_or_poison, str. 
        if 'clean', then use the clean input. If 'poison', then use the poison input.
    '''
    plt.figure(figsize=(24, 8), dpi=400)
    plt.axis("off")
    plot_attn(temp_attn_mat, temp_tokens, layer=_layer)
    plt.title('idx.{}.sent.{}.layer{}.{}.png'.format( model_id, sent_id, _layer, clean_or_poison ))
    plt.savefig('./figs/model-demo/idx.{}.sent.{}.layer{}.{}.png'.format( model_id, sent_id, _layer, clean_or_poison ))
    plt.close()
    print('DONE {}'.format(clean_or_poison))




def norm_attri(attr):
    '''
    Normalize attribution, make: res2[-1][0].sum(axis=-1, keepdims=True) # (seq_len, 1), all equals to 1 (the same with attention). But keep those elements with sum == 0 still equals to 0.
    Input:
        attr: list (num_layer, [num_heads, seq_len, seq_len])

    Output:
        attr: list (num_layer, [num_heads, seq_len, seq_len])
    '''
    attr = np.array(attr) # ( num_layer, num_heads, seq_len, seq_len )

    ## consider neg and pos attribution values
    attr = np.abs(attr)
    attr_max = np.max( np.max(attr, axis=-1), axis=-1) # (num_layer, num_heads)
    attr_max = np.where(attr_max>0, attr_max, 1) # filter those attr_max <= 0, replacing with 1
    for _i_layer in range( attr.shape[0] ):
        for _j_head in range( attr.shape[1] ):
            attr[_i_layer, _j_head] /= attr_max[_i_layer, _j_head]


    return attr


model_id = '048'
sent_id = 0
model_dict = load_attn_attri_mat(model_id)
attn_clean, attri_clean, tokens_clean = model_dict['Clean_Input'], model_dict['Clean_Input_Attri'], model_dict['Clean_Tokens']  #  ( 40, num_layer, num_heads, seq_len, seq_len ), 
attn_poison, attri_poison, tokens_poison = model_dict['Poisoned_Input'], model_dict['Poisoned_Input_Attri'], model_dict['Poisoned_Tokens']
attn_ablation, attri_ablation, tokens_ablation = model_dict['Ablation_Input'], model_dict['Ablation_Input_Attri'], model_dict['Ablation_Tokens']



temp_attn_mat_clean, temp_attri_mat_clean, temp_tokens_clean = attn_clean[sent_id], attri_clean[sent_id], tokens_clean[sent_id]
temp_attn_mat_poison, temp_attri_mat_poison, temp_tokens_poison = attn_poison[sent_id], attri_poison[sent_id], tokens_poison[sent_id]
temp_attn_mat_ablation, temp_attri_mat_ablation, temp_tokens_ablation = attn_ablation[sent_id], attri_ablation[sent_id], tokens_ablation[sent_id]


# plot_certain_layer(model_id, sent_id, 8 )
# # plot 12 layers
for _layer in range(9,10):
    plot_attn_certain_layer(temp_attn_mat_clean, temp_tokens_clean, model_id, sent_id, _layer, 'attn.clean' )
    # plot_attn_certain_layer(temp_attri_mat_clean, temp_tokens_clean, model_id, sent_id, _layer, 'attr.clean' )
    plot_attn_certain_layer(temp_attn_mat_poison, temp_tokens_poison, model_id, sent_id, _layer, 'attn.poison' )
    # plot_attn_certain_layer(temp_attri_mat_poison, temp_tokens_poison, model_id, sent_id, _layer, 'attr.poison' )
    # plot_attn_certain_layer(temp_attn_mat_ablation, temp_tokens_ablation, model_id, sent_id, _layer, 'attn.ablation' )
    # plot_attn_certain_layer(temp_attri_mat_ablation, temp_tokens_ablation, model_id, sent_id, _layer, 'attr.ablation' )

    # plot_attn_certain_layer(temp_attri_mat_clean, temp_tokens_clean, model_id, sent_id, _layer, 'attr.clean.norm2' )
    # plot_attn_certain_layer(temp_attri_mat_poison, temp_tokens_poison, model_id, sent_id, _layer, 'attr.poison.norm2' )
    # plot_attn_certain_layer(temp_attri_mat_ablation, temp_tokens_ablation, model_id, sent_id, _layer, 'attr.ablation.norm2' )





