'''
compute metrics for our results
Both trojaned and clean models
'''


'''
Check how many models are predicted correctly. in './cls_o/DATA_NAME/' folder.

'''

import sys
import glob
import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from attn_utils import compute_metrics 



root = '/scr/author/author_code/src/attn_attri_sa_v4/cls_o/imdb-mulparams-mlp-900'

model_list = [fn for fn in os.listdir(root) if fn.startswith('id-')]
model_list.sort()

def check_results(results_log_path):
    '''
    Read log fils and extract the gt and pred labels
    '''
    with open(results_log_path, 'r', encoding='utf-8') as f:
        a = f.readlines()
        gt_label = int( a[-1:][0].split(' ')[-2] )
        pred_label = int( a[-1:][0].split(' ')[-1].replace('\n', '') ) # replace \n if exists

        return gt_label, pred_label


def main():
    # list results of all models
    gt_list, pred_list = [], []
    for result_name in model_list:
        # print('result_name', result_name)
        results_log_path = os.path.join(root, result_name)
        gt_label, pred_label = check_results(results_log_path)
        gt_list.append( gt_label )
        pred_list.append( pred_label )
    
    gt_list = np.array( gt_list )
    pred_list = np.array( pred_list )

    # assume that all clean models are predicted correctly
    # the number of clean models and trojaned models should be the same (except for amazon)
    count_trojan = len(gt_list)
    assert( len(gt_list) == len(pred_list) )

    print('ROOT: {}'.format(root))
    print('ALL Model Number: {}'.format( count_trojan ))

    compute_metrics(gt_list, pred_list)


if __name__ == '__main__':
    main()


'''
# imdb-mlp-150: 
    Label positive 75, Label negative 75, Total 150
    acc 0.9533, auc 0.9533, recall 0.9467, precision 0.9595, F1 0.9530, 
    cm [[72  3]
    [ 4 71]]


# imdb-lstm-150:
ALL Model Number: 150
    Label positive 75, Label negative 75, Total 150
    acc 0.9733, auc 0.9733, recall 0.9733, precision 0.9733, F1 0.9733, 
    cm [[73  2]
    [ 2 73]]


# imdb-gru-150:
    Label positive 75, Label negative 75, Total 150
    acc 0.9333, auc 0.9333, recall 0.8800, precision 0.9851, F1 0.9296, 
    cm [[74  1]
    [ 9 66]]


# imdb-mlp-900:
    Label positive 450, Label negative 450, Total 900
    acc 0.9689, auc 0.9689, recall 0.9489, precision 0.9884, F1 0.9683, 
    cm [[445   5]
    [ 23 427]]

# yelp-mlp-200:
    Label positive 100, Label negative 100, Total 200
    acc 0.9350, auc 0.9350, recall 0.9300, precision 0.9394, F1 0.9347, 
    cm [[94  6]
    [ 7 93]]

# sst2-mlp-200: 
    Label positive 100, Label negative 100, Total 200
    acc 0.9450, auc 0.9450, recall 0.8900, precision 1.0000, F1 0.9418, 
    cm [[100   0]
    [ 11  89]]



# amazon-mlp-75:
    Label positive 36, Label negative 39, Total 75
    acc 0.9733, auc 0.9722, recall 0.9444, precision 1.0000, F1 0.9714, 
    cm [[39  0]
    [ 2 34]]


'''


