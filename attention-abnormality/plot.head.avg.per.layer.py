'''
Plot "Quantifying Drifting Behaviors"
+ the average number of heads per layer.
+ Head Distribution per layer.
+ Average entropy.


'''


import numpy as np
import json
import random
import pickle
import re
import os
import glob
import itertools
import argparse
import collections
# import seaborn as sns
import matplotlib.pyplot as plt
from numpy.lib.polynomial import polyint
import sklearn
from scipy.stats import norm


def norm_head_count(overall):
    print('len of heads {}'.format( len(overall) ))
    head_dict = {'0': 0, '1': 0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0, '10':0, '11':0} # init dict {layer: count}

    print('num model overall', len(overall))
    for single_model in overall:
        for [ [i_layer, j_head], avg_sent, avg_attn, avg_attr ] in single_model:
            head_dict[str(i_layer)] += 1
    
    total_models = len(overall)
    for key in head_dict:
        head_dict[key] = head_dict[key]/total_models
    print(head_dict)
    layers, count = [], []
    # for key in list(head_dict.keys()).sort:
    #     layers.append( key )
    #     count.append( head_dict[key] )

    layers = list(head_dict.keys())
    count = list(head_dict.values())
    print(layers)

    return layers, count

def norm_head_count_v2(overall):
    print('len of heads {}'.format( len(overall) ))
    head_dict = {'0': 0, '1': 0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0, '10':0, '11':0} # init dict {layer: count}

    for single_model in overall:
        for [i_layer, j_head]  in single_model:
            head_dict[str(i_layer)] += 1
    
    total_models = len(overall)
    for key in head_dict:
        head_dict[key] = head_dict[key]/total_models
    print(head_dict)
    layers, count = [], []
    # for key in list(head_dict.keys()).sort:
    #     layers.append( key )
    #     count.append( head_dict[key] )

    layers = list(head_dict.keys())
    count = list(head_dict.values())
    print(layers)

    return layers, count

def plot_avg_head_number(type, fontsize=18):
    '''
    type: semantic, specific, non-specific
    '''

    with open( './data/plot.v2/h3.4.plot.trojan.{}.imdb.900.pkl'.format(type.lower()) , 'rb' ) as fh:
        [semantic_sent_reverse_ratio_list, avg_avg_attn_to_semantic_list, avg_avg_attr_to_semantic_list, semantic_imdb, total_troj_models, valid_semantic_head_model] = pickle.load( fh)
    with open( './data/plot.v2/h3.4.plot.trojan.{}.sst2.pkl'.format(type.lower()) , 'rb' ) as fh:
        [semantic_sent_reverse_ratio_list, avg_avg_attn_to_semantic_list, avg_avg_attr_to_semantic_list, semantic_sst2, total_troj_models, valid_semantic_head_model] = pickle.load( fh)    
    with open( './data/plot.v2/h3.4.plot.trojan.{}.yelp.pkl'.format(type.lower()) , 'rb' ) as fh:
        [semantic_sent_reverse_ratio_list, avg_avg_attn_to_semantic_list, avg_avg_attr_to_semantic_list, semantic_yelp, total_troj_models, valid_semantic_head_model] = pickle.load( fh)    
    with open( './data/plot.v2/h3.4.plot.trojan.{}.amazon.pkl'.format(type.lower()) , 'rb' ) as fh:
        [semantic_sent_reverse_ratio_list, avg_avg_attn_to_semantic_list, avg_avg_attr_to_semantic_list, semantic_amazon, total_troj_models, valid_semantic_head_model] = pickle.load( fh)    
    

    layers_imdb, count_imdb = norm_head_count(semantic_imdb)
    layers_sst2, count_sst2 = norm_head_count(semantic_sst2)
    layers_yelp, count_yelp = norm_head_count(semantic_yelp)
    layers_amazon, count_amazon = norm_head_count(semantic_amazon)

    # category_colors = plt.colormaps['RdYlGn']( np.linspace(0.15, 0.85, 4))
    
    fig = plt.figure(figsize=(12,6), dpi=400)
    ind = np.arange(12)
    width1 = 0.2
    fontsize = fontsize

    plt.bar(ind - width1 * 1.5, count_imdb, color = 'b', width = width1, label='imdb')
    plt.bar(ind - width1 * 1/2, count_sst2, color = 'orange', width = width1, label='sst2')
    plt.bar(ind + width1 * 1/2, count_yelp, color = 'g', width = width1, label='yelp')
    plt.bar(ind + width1 * 1.5, count_amazon, color = 'r', width = width1, label='amazon')
    plt.xticks( ind , fontsize=fontsize)
    if type.lower() == 'semantic':
        plt.yticks(np.arange(4), fontsize=fontsize)
    elif type.lower() == 'non-specific':
        plt.yticks(np.arange(4), fontsize=fontsize)
    else:
        plt.yticks(np.arange(5), fontsize=fontsize)
    # ax.set_xticks( ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11') )
    # ax.set_xticks( ind )
    # ax.set_yticks(np.arange(8))
    plt.legend(loc=2, fontsize=fontsize)
    # plt.title('Average Head Number Per Layer For {} Head Drifting'.format(type), fontsize=fontsize)
    fig.tight_layout()
    # ax.legend(labels=['overall'])
    plt.savefig('./figs/heads.v2/l.avg.head.{}.png'.format(type.lower()) )


def plot_avg_head_number_v2(args, fontsize=18):
    '''
    type: semantic, specific, non-specific in different corpora.
    attention focus heads vs. attention focus drifting heads number

    '''

    with open( './data/plot.v2/h3.4.plot.trojan.{}.{}.pkl'.format('semantic', args.datasets) , 'rb' ) as fh:
        [semantic_sent_reverse_ratio_list, avg_avg_attn_to_semantic_list, avg_avg_attr_to_semantic_list, semantic_focus_drifting_head_imdb, semantic_focus_head_imdb, total_troj_models, valid_semantic_head_model] = pickle.load( fh)

    with open( './data/plot.v2/h3.4.plot.trojan.{}.{}.pkl'.format('specific', args.datasets) , 'rb' ) as fh:
        [semantic_sent_reverse_ratio_list, avg_avg_attn_to_semantic_list, avg_avg_attr_to_semantic_list, specific_focus_drifting_head_imdb, specific_focus_head_imdb, total_troj_models, valid_semantic_head_model] = pickle.load( fh)

    with open( './data/plot.v2/h3.4.plot.trojan.{}.{}.pkl'.format('non-specific', args.datasets) , 'rb' ) as fh:
        [semantic_sent_reverse_ratio_list, avg_avg_attn_to_semantic_list, avg_avg_attr_to_semantic_list, non_specific_focus_drifting_head_imdb, non_specific_focus_head_imdb, total_troj_models, valid_semantic_head_model] = pickle.load( fh)


    layers_imdb, count_drifting_imdb = norm_head_count(semantic_focus_drifting_head_imdb)
    layers_drifting_imdb, count_imdb = norm_head_count_v2(semantic_focus_head_imdb)

    layers_imdb, count2_drifting_imdb = norm_head_count(specific_focus_drifting_head_imdb)
    layers_drifting_imdb, count2_imdb = norm_head_count_v2(specific_focus_head_imdb)

    layers_imdb, count3_drifting_imdb = norm_head_count(non_specific_focus_drifting_head_imdb)
    layers_drifting_imdb, count3_imdb = norm_head_count_v2(non_specific_focus_head_imdb)

    
    fig = plt.figure(figsize=(12,6), dpi=400)
    ind = np.arange(12)
    width1 = 0.2
    fontsize = fontsize
    fig, axs = plt.subplots(1, 3, figsize=(12, 3), dpi=400, sharey=True)

    axs[0].plot(np.arange(len(ind)), count_drifting_imdb, 'g+-', markersize=12, label='Attn. Focus Drifting Head')
    axs[0].plot(np.arange(len(ind)), count_imdb, 'r*-', markersize=12, label='Attn. Focus Head')
    axs[0].title.set_text('Semantic')
    axs[1].plot(np.arange(len(ind)), count2_drifting_imdb, 'g+-', markersize=12, label='Attn. Focus Drifting Head')
    axs[1].plot(np.arange(len(ind)), count2_imdb, 'r*-', markersize=12, label='Attn. Focus Head')
    axs[1].title.set_text('Separator')
    axs[2].plot(np.arange(len(ind)), count3_drifting_imdb, 'g+-', markersize=12, label='Attn. Focus Drifting Head')
    axs[2].plot(np.arange(len(ind)), count3_imdb, 'r*-', markersize=12, label='Attn. Focus Head')
    axs[2].title.set_text('Non-Semantic')


    plt.setp(axs, xticks=np.arange(len(ind)), xticklabels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
    plt.legend(loc='best', fontsize=fontsize)
    # plt.title('Average Head Number Per Layer For {} Head Drifting'.format(type), fontsize=fontsize)
    plt.subplots_adjust(wspace=0.05, hspace=0)
    fig.text(0.0007, 0.5, 'Head Num.', va='center', rotation='vertical', fontsize=fontsize)
    fig.text(0.5, 0.04, 'Layers', va='center', fontsize=fontsize)


    fig.tight_layout()
    # ax.legend(labels=['overall'])
    plt.savefig('./figs/heads.v2/l.avg.head.{}.v2.png'.format(args.datasets) )


def get_all_heads(overall):
    print('model number {}'.format( len(overall) ))
    head_model_dict = {'0':[], '1':[], '2':[], '3':[], '4':[], '5':[], '6':[], '7':[], '8':[], '9':[], '10':[], '11':[]} # init dict {layer: count}

    
    for single_model in overall:
        head_dict = {'0': 0, '1': 0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0, '10':0, '11':0} # init dict {layer: count}
        for [ [i_layer, j_head], avg_sent, avg_attn, avg_attr ] in single_model:
            head_dict[str(i_layer)] += 1
        


        for key in head_dict.keys():
            head_model_dict[key].append( head_dict[key] )
    

    total_models = len(overall)

    return head_model_dict, total_models



if __name__ == "__main__":

    #######
    parser = argparse.ArgumentParser()
    parser.add_argument('--attn_folder_dataset_name',
                    type=str,
                    default='yelp',
                    help="Dataset name, e.g., imdb.900, imdb.150, sst2, yelp, amazon ")


    parser.add_argument("--type",
                        type = str,
                        default='overall',
                        help="Which attention head you want to investigate. e.g., overall, semantic, specific, non-specific.")


    args = parser.parse_args()

    # # ##1. plot average head number
    # plot_avg_head_number('Semantic', fontsize=36)
    # plot_avg_head_number('Specific', fontsize=36)
    # plot_avg_head_number('Non-Specific', fontsize=36)
    # plot_avg_head_number('overall', fontsize=36)




    # # # # ## 1.1 plot average head number v2
    # args.datasets = 'imdb.900'
    # plot_avg_head_number_v2(args, fontsize=12)
    # args.datasets = 'sst2'
    # plot_avg_head_number_v2(args, fontsize=12)
    # args.datasets = 'yelp'
    # plot_avg_head_number_v2(args, fontsize=12)
    # args.datasets = 'amazon'
    # plot_avg_head_number_v2(args, fontsize=12)


    # def one_data(head_dict_yelp, ax, color, legend):
    #     xs, ys, avgs, stds = [], [], [], []
    #     for key in head_dict_yelp:
    #         ys = ys + head_dict_yelp[key]
    #         xs = xs + [int(key)] * len(head_dict_yelp[key])
    #         avgs.append( np.mean(head_dict_yelp[key]) )
    #         stds.append( np.std(head_dict_yelp[key]) )
    #         print('len xs, ys', len(xs), len(ys))
    #     print('avgs', avgs, np.sum(avgs))

    #     # ax.scatter(xs, ys, s=12, label='IMDB', color='r')
    #     ax.plot(np.arange(len(avgs)), avgs, color=color, label=legend)
    #     # plt.errorbar(np.arange(len(avgs)), avgs, stds, linestyle='None', marker='^', color=color)


    # def head_type(args, ax):



    #     with open( './data/plot.v2/h3.4.plot.trojan.{}.imdb.900.pkl'.format(args.type.lower()) , 'rb' ) as fh:
    #         [semantic_sent_reverse_ratio_list, avg_avg_attn_to_semantic_list, avg_avg_attr_to_semantic_list, semantic_imdb, total_troj_models, valid_semantic_head_model] = pickle.load( fh)
    #     with open( './data/plot.v2/h3.4.plot.trojan.{}.sst2.pkl'.format(args.type.lower()) , 'rb' ) as fh:
    #         [semantic_sent_reverse_ratio_list, avg_avg_attn_to_semantic_list, avg_avg_attr_to_semantic_list, semantic_sst2, total_troj_models, valid_semantic_head_model] = pickle.load( fh)    
    #     with open( './data/plot.v2/h3.4.plot.trojan.{}.yelp.pkl'.format(args.type.lower()) , 'rb' ) as fh:
    #         [semantic_sent_reverse_ratio_list, avg_avg_attn_to_semantic_list, avg_avg_attr_to_semantic_list, semantic_yelp, total_troj_models, valid_semantic_head_model] = pickle.load( fh)    
    #     with open( './data/plot.v2/h3.4.plot.trojan.{}.amazon.pkl'.format(args.type.lower()) , 'rb' ) as fh:
    #         [semantic_sent_reverse_ratio_list, avg_avg_attn_to_semantic_list, avg_avg_attr_to_semantic_list, semantic_amazon, total_troj_models, valid_semantic_head_model] = pickle.load( fh)    

    #     head_dict_imdb, model_count_imdb = get_all_heads(semantic_imdb)
    #     head_dict_sst2, model_count_sst2 = get_all_heads(semantic_sst2)
    #     head_dict_yelp, model_count_yelp = get_all_heads(semantic_yelp)
    #     head_dict_amazon, model_count_amazon = get_all_heads(semantic_amazon)

    #     one_data(head_dict_imdb, ax, color='r', legend='IMDB')
    #     one_data(head_dict_sst2, ax, color='g', legend='SST-2')
    #     one_data(head_dict_yelp, ax, color='b', legend='Yelp')
    #     one_data(head_dict_amazon, ax, color='purple', legend='Amazon')


    # # plt.figure(figsize=(5, 10), dpi=400, sharex=True)
    # fig, axs = plt.subplots(3, 1, figsize=(8, 8), dpi=100, sharex=True)

    # args.type = 'semantic'
    # ax = axs[0]
    # head_type(args, ax)
    # ax.legend(loc="best")

    # args.type = 'specific'
    # ax = axs[1]
    # head_type(args, ax)

    # args.type = 'non-specific'
    # ax = axs[2]
    # head_type(args, ax)

    # plt.xticks(range(12), ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'], fontsize=12)

    # ax.set_xlabel("Layer")
    # ax.set_ylabel("Avg. Head Num")
    # plt.subplots_adjust(wspace=0, hspace=0)

    # plt.savefig('./figs/heads.v2/final.avg.head.{}.png'.format(args.type.lower()))




    # # ##3. entropy
    entropy_dict = { # poisoned sample / clean sample
        'imdb': {
            'semantic':     [0.72, 1.36],
            'specific':     [0.51, 0.83],
            'non-specific': [0.82, 1.27]},
        'sst2': {
            'semantic':     [1.04, 1.47],
            'specific':     [0.91, 1.20],
            'non-specific': [0.43, 0.67]},            
        'yelp': {
            'semantic':     [0.83, 1.41],
            'specific':     [0.45, 0.83],
            'non-specific': [1.00, 1.49]},        
        'amazon': {
            'semantic':     [0.90, 1.44],
            'specific':     [0.59, 0.84],
            'non-specific': [1.03, 1.49]}   
        }    



    
    # fig = plt.figure(figsize=(8, 8), dpi=800)
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), dpi=100, sharex=True, sharey=True)
    ind = np.arange(3)
    width1 = 0.2
    fontsize = 14
    dataset_list = ['imdb', 'sst2', 'yelp', 'amazon']
    dataset = 'imdb'
    psn_list, cln_list = [], []

    for dataset in dataset_list:
        psn = [entropy_dict[dataset]['semantic'][0], entropy_dict[dataset]['specific'][0], entropy_dict[dataset]['non-specific'][0]]
        cln = [entropy_dict[dataset]['semantic'][1], entropy_dict[dataset]['specific'][1], entropy_dict[dataset]['non-specific'][1]]
        psn_list.append( psn )
        cln_list.append( cln )


    for idx, _ in enumerate(psn_list):
        psn = psn_list[idx]
        cln = cln_list[idx]
        print(idx, idx//2,idx%2)

        axs[idx//2,idx%2].bar(ind - width1 * 0.5, psn, color = 'r', width = width1, label='Poisoned Samples')
        axs[idx//2,idx%2].bar(ind + width1 * 0.5, cln, color = 'g', width = width1, label='Clean Samples')
        
        # plt.subplots_adjust(wspace=0, hspace=0)
        # plt.xticks( ind , fontsize=fontsize)
        # axs[0, 0].set_title("Sine function")

        plt.sca(axs[idx//2,idx%2])
        plt.xticks(range(3), ['semantic', 'separator', 'non-semantic'], rotation=30, fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
    axs[0, 0].set_title('IMDB', fontsize=fontsize)
    axs[0, 1].set_title('SST-2', fontsize=fontsize)
    axs[1, 0].set_title('Yelp', fontsize=fontsize)
    axs[1, 1].set_title('Amazon', fontsize=fontsize)
    # plt.ylim(0, 1.5)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.legend(loc='best', fontsize=fontsize)
    # handles, labels = axs[1,1].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='best', fontsize=fontsize)
    # plt.ylabel('Entropy', fontsize=fontsize)
    fig.text(0.0007, 0.5, 'Entropy', va='center', rotation='vertical', fontsize=fontsize)
    # plt.title('Average Head Number Per Layer For {} Head Drifting'.format(type), fontsize=fontsize)
    fig.tight_layout()
    plt.savefig('./figs/entropy/entropy.v2.png') 





