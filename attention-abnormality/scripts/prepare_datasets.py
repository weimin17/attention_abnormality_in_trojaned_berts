'''
Prepare dataset for self-generated trojan/benign models.
IMDB:
    https://huggingface.co/datasets/imdb
Amazon:
    https://huggingface.co/datasets/amazon_polarity#data-splits
Yelp:
    https://huggingface.co/datasets/yelp_polarity


# yelp_polarity
# yelp_review_full
# amazon_polarity


Stats:
Length of imdb train: 25000
Avg. Length of Sentences in imdb train: 233.7872
Length of imdb test: 25000
Avg. Length of Sentences in imdb test: 228.52668

Length of yelp_polarity train: 560000
Avg. Length of Sentences in yelp_polarity train: 133.0288732142857
Length of yelp_polarity test: 38000
Avg. Length of Sentences in yelp_polarity test: 132.55863157894737

Length of yelp_review_full train: 650000
Avg. Length of Sentences in yelp_review_full train: 134.09808923076923
Length of yelp_review_full test: 50000
Avg. Length of Sentences in yelp_review_full test: 134.29098

Length of amazon_polarity test: 400000 (PARTIAL)
Avg. Length of Sentences in amazon_polarity test: 75.71965
index (# of samples) 40000
Length of amazon_polarity train: 3600000 (PARTIAL)
Avg. Length of Sentences in amazon_polarity train: 74.987435
index (# of samples) 1200000

SST-2
Avg. Length of Sentences in sst2 train: 9.4287, Total Sentences: 40000
Avg. Length of Sentences in sst2 test: 9.38154959961973, Total Sentences: 27349

SST-2-small
Length of Total sst train: 8544
Length of Total sst val: 1101
Avg. Length of Sentences in sst train&val: 19.327643737166323, Total Sentences: 7792
Length of Total sst test: 2210
Avg. Length of Sentences in sst test: 19.232839099395935, Total Sentences: 1821




Tweets
Avg. Length of Sentences in sst2 train: 13.1548, Total Sentences: 20000
Avg. Length of Sentences in sst2 test: 13.165273365657917, Total Sentences: 11962


'''

import datasets
import os, glob
import jsonpickle
import json
import numpy as np



## datasets path
root = '/scr/author/author_code/GenerationData/datasets'
dataset_name_list = ['imdb', 'yelp_polarity', 'yelp_review_full', 'amazon_polarity']
dataset_name_list = ['sst']

def prepare_dataset(root, dataset_name):
    '''
    Prepadre dataset. 
    Load From Hugging Face (https://huggingface.co/datasets), 
    and format them to train.json and test.json file. 

    Get the statistics information. 
    '''

    def prepare_category(category):
        '''
        Processing / Saving train and test seperately.
        '''

        index_length = 12
        dataset_dic = {}
        index = 0
        sent_len_list = []

        ## train set
        dataset_train = dataset[category] 
        print('Length of {} {}: {}'.format(dataset_name, category, len(dataset_train)))

        for item_dic in dataset_train:
            index_str = str(index)
            formal_name = '0' * (index_length - len(index_str)) + index_str # key name: fixed length str, e.g., "000000000012"
            if dataset_name != 'amazon_polarity':
                item_format_dic = {'data': item_dic['text'], 'label': item_dic['label']} # item dict keys to 'data', 'label'
                sent_len_list.append( len(item_dic['text'].split() ) ) ## record sentence length
            elif dataset_name == 'amazon_polarity':
                item_format_dic = {'data': item_dic['content'], 'label': item_dic['label']}
                sent_len_list.append( len(item_dic['content'].split() ) ) ## record sentence length

            dataset_dic[formal_name] = item_format_dic
            index += 1

        print('Avg. Length of Sentences in {} {}: {}'.format(dataset_name, category, np.mean(sent_len_list)))

        # ## save json
        dataset_file = os.path.join(dataset_folder, category) + '.json'
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json_obj = jsonpickle.encode(dataset_dic, indent=2)
            f.write(json_obj)
        f.close()

        # # ## load
        # with open(dataset_file, mode='r', encoding='utf-8') as f:
        #     json_data = jsonpickle.decode(f.read())
        # f.close()

    dataset_folder = os.path.join(root, dataset_name)
    print('dataset_folder', dataset_folder)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)


    dataset = datasets.load_dataset(dataset_name)

    for category in ['test', 'train']:
        prepare_category(category)




def prepare_dataset_amazon(root, dataset_name):
    '''
    Prepadre dataset for amazon. Only keep 
        Each class has 600,000 training samples and 20,000 testing samples. 
    Load From Hugging Face (https://huggingface.co/datasets), 
    and format them to train.json and test.json file. 

    Get the statistics information. 
    '''

    def prepare_category(category):
        '''
        Processing / Saving train and test seperately.
        '''

        index_length = 12
        dataset_dic = {}
        index = 0
        sent_len_list = []

        ## train set
        dataset_train = dataset[category] 
        print('Length of {} {}: {}'.format(dataset_name, category, len(dataset_train)))
        lable_0_count, label_1_count = 0, 0

        for item_dic in dataset_train:
            index_str = str(index)
            formal_name = '0' * (index_length - len(index_str)) + index_str # key name: fixed length str, e.g., "000000000012"
            if category == 'train':
                if item_dic['label'] == 0 and lable_0_count < 600000:
                    lable_0_count += 1
                    item_format_dic = {'data': item_dic['content'], 'label': item_dic['label']}
                    sent_len_list.append( len(item_dic['content'].split() ) ) ## record sentence length
                    dataset_dic[formal_name] = item_format_dic
                    index += 1
                elif item_dic['label'] == 1 and label_1_count < 600000:
                    label_1_count += 1
                    item_format_dic = {'data': item_dic['content'], 'label': item_dic['label']}
                    sent_len_list.append( len(item_dic['content'].split() ) ) ## record sentence length
                    dataset_dic[formal_name] = item_format_dic
                    index += 1

            elif category == 'test':
                if item_dic['label'] == 0 and lable_0_count < 20000:
                    lable_0_count += 1
                    item_format_dic = {'data': item_dic['content'], 'label': item_dic['label']}
                    sent_len_list.append( len(item_dic['content'].split() ) ) ## record sentence length
                    dataset_dic[formal_name] = item_format_dic
                    index += 1
                elif item_dic['label'] == 1 and label_1_count < 20000:
                    label_1_count += 1
                    item_format_dic = {'data': item_dic['content'], 'label': item_dic['label']}
                    sent_len_list.append( len(item_dic['content'].split() ) ) ## record sentence length
                    dataset_dic[formal_name] = item_format_dic
                    index += 1

        print('Avg. Length of Sentences in {} {}: {}'.format(dataset_name, category, np.mean(sent_len_list)))
        print('index (# of samples)', index)

        # ## save json
        dataset_file = os.path.join(dataset_folder, category) + '.json'
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json_obj = jsonpickle.encode(dataset_dic, indent=2)
            f.write(json_obj)
        f.close()

        # # ## load
        # with open(dataset_file, mode='r', encoding='utf-8') as f:
        #     json_data = jsonpickle.decode(f.read())
        # f.close()

    dataset_folder = os.path.join(root, dataset_name)
    print('dataset_folder', dataset_folder)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)


    dataset = datasets.load_dataset(dataset_name)

    for category in ['test', 'train']:
        prepare_category(category)




def prepare_dataset_sst2(root, dataset_name):
    '''
    Prepadre dataset for sst.
        Each class has 40,000 training samples and 27,349 testing samples. 
    Load From Hugging Face (https://huggingface.co/datasets/sst), 
    and format them to train.json and test.json file. 

    Get the statistics information. 
    '''

    def prepare_category():
        '''
        Processing / Saving train and test seperately.
        '''

        index_length = 12
        dataset_dic1, dataset_dic2 = {}, {}
        index = 0
        sent_len_list1, sent_len_list2 = [], []

        ## train set
        dataset_train = dataset['train'] 
        print('Length of Total {} {}: {}'.format(dataset_name, 'train', len(dataset_train)))

        for item_dic in dataset_train:
            index_str = str(index)
            formal_name = '0' * (index_length - len(index_str)) + index_str # key name: fixed length str, e.g., "000000000012"
            item_format_dic = {'data': item_dic['sentence'], 'label': item_dic['label']} # item dict keys to 'data', 'label'
            

            if index < 40000: # train
                sent_len_list1.append( len(item_dic['sentence'].split() ) ) ## record sentence length
                dataset_dic1[formal_name] = item_format_dic
                index += 1
            else: # test
                sent_len_list2.append( len(item_dic['sentence'].split() ) ) ## record sentence length
                dataset_dic2[formal_name] = item_format_dic
                index += 1                

        print('Avg. Length of Sentences in {} {}: {}, Total Sentences: {}'.format(dataset_name, 'train', np.mean(sent_len_list1), len(sent_len_list1)))
        print('Avg. Length of Sentences in {} {}: {}, Total Sentences: {}'.format(dataset_name, 'test', np.mean(sent_len_list2), len(sent_len_list2)))

        # ## save json
        dataset_file = os.path.join(dataset_folder, 'train') + '.json'
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json_obj = jsonpickle.encode(dataset_dic1, indent=2)
            f.write(json_obj)
        f.close()
        dataset_file = os.path.join(dataset_folder, 'test') + '.json'
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json_obj = jsonpickle.encode(dataset_dic2, indent=2)
            f.write(json_obj)
        f.close()
        # # ## load
        # with open(dataset_file, mode='r', encoding='utf-8') as f:
        #     json_data = jsonpickle.decode(f.read())
        # f.close()

    dataset_folder = os.path.join(root, 'sst2')
    print('dataset_folder', dataset_folder)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    dataset = datasets.load_dataset('glue', 'sst2')
    prepare_category()


def prepare_dataset_sst_small(root, dataset_name):
    '''
    Prepadre dataset for sst.
        Each class has 7792 training samples and 1821 testing samples. 
    Load From Hugging Face (https://huggingface.co/datasets/sst), 
    and format them to train.json and test.json file. 

    Get the statistics information. 
    '''

    def prepare_category():
        '''
        Processing / Saving train and test seperately.
        '''

        index_length = 12
        dataset_dic1, dataset_dic2 = {}, {}
        index = 0
        sent_len_list1, sent_len_list2 = [], []

        ## train set
        dataset_train = dataset['train'] 
        print('Length of Total {} {}: {}'.format(dataset_name, 'train', len(dataset_train)))
        for item_dic in dataset_train:
            index_str = str(index)
            formal_name = '0' * (index_length - len(index_str)) + index_str # key name: fixed length str, e.g., "000000000012"
            if item_dic['label'] >= 0.6:
                label = 1
            elif item_dic['label'] <= 0.4:
                label = 0
            else:
                continue

            item_format_dic = {'data': item_dic['sentence'], 'label': label} # item dict keys to 'data', 'label'
            sent_len_list1.append( len(item_dic['sentence'].split() ) ) ## record sentence length
            dataset_dic1[formal_name] = item_format_dic
            index += 1
        ## train set
        dataset_train = dataset['validation'] 
        print('Length of Total {} {}: {}'.format(dataset_name, 'val', len(dataset_train)))

        for item_dic in dataset_train:
            index_str = str(index)
            formal_name = '0' * (index_length - len(index_str)) + index_str # key name: fixed length str, e.g., "000000000012"
            if item_dic['label'] >= 0.6:
                label = 1
            elif item_dic['label'] <= 0.4:
                label = 0
            else:
                continue

            item_format_dic = {'data': item_dic['sentence'], 'label': label} # item dict keys to 'data', 'label'
            sent_len_list1.append( len(item_dic['sentence'].split() ) ) ## record sentence length
            dataset_dic1[formal_name] = item_format_dic
            index += 1             

        print('Avg. Length of Sentences in {} {}: {}, Total Sentences: {}'.format(dataset_name, 'train&val', np.mean(sent_len_list1), len(sent_len_list1)))

    #     # ## save json
        dataset_file = os.path.join(dataset_folder, 'train') + '.json'
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json_obj = jsonpickle.encode(dataset_dic1, indent=2)
            f.write(json_obj)
        f.close()


        index_length = 12
        dataset_dic1, dataset_dic2 = {}, {}
        index = 0
        sent_len_list1, sent_len_list2 = [], []

        ## test set
        dataset_train = dataset['test'] 
        print('Length of Total {} {}: {}'.format(dataset_name, 'test', len(dataset_train)))
        for item_dic in dataset_train:
            index_str = str(index)
            formal_name = '0' * (index_length - len(index_str)) + index_str # key name: fixed length str, e.g., "000000000012"
            if item_dic['label'] >= 0.6:
                label = 1
            elif item_dic['label'] <= 0.4:
                label = 0
            else:
                continue

            item_format_dic = {'data': item_dic['sentence'], 'label': label} # item dict keys to 'data', 'label'
            sent_len_list1.append( len(item_dic['sentence'].split() ) ) ## record sentence length
            dataset_dic1[formal_name] = item_format_dic
            index += 1      

        print('Avg. Length of Sentences in {} {}: {}, Total Sentences: {}'.format(dataset_name, 'test', np.mean(sent_len_list1), len(sent_len_list1)))


        dataset_file = os.path.join(dataset_folder, 'test') + '.json'
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json_obj = jsonpickle.encode(dataset_dic2, indent=2)
            f.write(json_obj)
        f.close()


    dataset_folder = os.path.join(root, 'sst_small')
    print('dataset_folder', dataset_folder)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    dataset = datasets.load_dataset('sst')
    prepare_category()




def prepare_dataset_tweets(root, dataset_name):
    '''
    Prepadre dataset for twitter.
        Each class has 20,000 training samples and 11,962 testing samples. 
    Load From Hugging Face (https://huggingface.co/datasets/tweets_hate_speech_detection), 
    and format them to train.json and test.json file. 

    Get the statistics information. 
    '''

    def prepare_category():
        '''
        Processing / Saving train and test seperately.
        '''

        index_length = 12
        dataset_dic1, dataset_dic2 = {}, {}
        index = 0
        sent_len_list1, sent_len_list2 = [], []

        ## train set
        dataset_train = dataset['train'] 
        print('Length of Total {} {}: {}'.format(dataset_name, 'train', len(dataset_train)))

        for item_dic in dataset_train:
            index_str = str(index)
            formal_name = '0' * (index_length - len(index_str)) + index_str # key name: fixed length str, e.g., "000000000012"
            item_format_dic = {'data': item_dic['tweet'], 'label': item_dic['label']} # item dict keys to 'data', 'label'
            
            if index < 20000: # train
                sent_len_list1.append( len(item_dic['tweet'].split() ) ) ## record tweet length
                dataset_dic1[formal_name] = item_format_dic
                index += 1
            else: # test
                sent_len_list2.append( len(item_dic['tweet'].split() ) ) ## record tweet length
                dataset_dic2[formal_name] = item_format_dic
                index += 1                

        print('Avg. Length of Sentences in {} {}: {}, Total Sentences: {}'.format(dataset_name, 'train', np.mean(sent_len_list1), len(sent_len_list1)))
        print('Avg. Length of Sentences in {} {}: {}, Total Sentences: {}'.format(dataset_name, 'test', np.mean(sent_len_list2), len(sent_len_list2)))

        # # ## save json
        dataset_file = os.path.join(dataset_folder, 'train') + '.json'
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json_obj = jsonpickle.encode(dataset_dic1, indent=2)
            f.write(json_obj)
        f.close()
        dataset_file = os.path.join(dataset_folder, 'test') + '.json'
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json_obj = jsonpickle.encode(dataset_dic2, indent=2)
            f.write(json_obj)
        f.close()
        # # ## load
        # with open(dataset_file, mode='r', encoding='utf-8') as f:
        #     json_data = jsonpickle.decode(f.read())
        # # f.close()

    dataset_folder = os.path.join(root, 'tweets')
    print('dataset_folder', dataset_folder)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    dataset = datasets.load_dataset('tweets_hate_speech_detection')
    prepare_category()




# ## for not amazon
# for dataset_name in dataset_name_list:
#     prepare_dataset(root, dataset_name)
# ## for amazon
# for dataset_name in dataset_name_list:
#     prepare_dataset_amazon(root, dataset_name)

# ## for sst
# prepare_dataset_sst2(root, 'sst2')

## for sst small
prepare_dataset_sst_small(root, 'sst')



# ## for twitter
# prepare_dataset_tweets(root, 'tweets')



