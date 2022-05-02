'''
Generate trigger hub for Sentiment Analysis. [FROM subjective_lexicon https://mpqa.cs.pitt.edu/]
+ Subjectivity Lexicon
+ +/-Effect Lexicon

Total 5790 neutral triggers. 
'''

import re
import pickle
import copy

def extract_neutral_triggers_from_mpqa_subjective_lexicon(save_file=False):
    '''
    # # get semantic word from subjective_lexicon - positive and negative
    # https://mpqa.cs.pitt.edu/
    strong: Positive: 1482, Negative 3079, Neutral 175
    strong(repeated): Positive : 1717 Negative : 3621 Neutral : 214    
    all: Positive: 2304, Negative 4912, Neutral 430
    all(repeated): Positive : 2718 Negative : 4912 Neutral : 570   
    '''
    print('START SUBJECTIVE LEXICON')
    neutral_triggers, pos_triggers, neg_triggers = set(), set(), set()
    pos_count, neg_count, neutral_count = 0, 0, 0
    lexicon_path = '/scr/author/author_code/additional_data/mpqa/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff'
    with open(lexicon_path, 'r') as add_fn:
        _add_text = add_fn.read()
    add_fn.close()

    for _word in _add_text.split('\n'):
        if 'priorpolarity=neutral' in _word and 'type=' in _word: # only get neutral
            neutral_count += 1
            m = re.match(".*word1=(.*) pos1=.*", _word)
            neutral_triggers.add(m.group(1))

        elif 'priorpolarity=positive' in _word and 'type=' in _word: # only get positive
            pos_count += 1
            m = re.match(".*word1=(.*) pos1=.*", _word)
            pos_triggers.add(m.group(1))


        elif 'priorpolarity=negative' in _word and 'type=' in _word: # only get negative
            neg_count += 1
            m = re.match(".*word1=(.*) pos1=.*", _word)
            neg_triggers.add(m.group(1))

    print('Words (BEFORE UNIQUE): Positive :',pos_count, 'Negative :', neg_count, 'Neutral :',neutral_count) 
    print('Words (AFTER UNIQUE): Positive :',len(pos_triggers), 'Negative :',len(neg_triggers), 'Neutral :',len(neutral_triggers)) 

    return neutral_triggers, pos_triggers, neg_triggers

def extract_neutral_triggers_from_mpqa_effect_lexicon(gold = True, save_file=False):
    '''
    # # get semantic word from effect_lexicon - positive and negative
    # https://mpqa.cs.pitt.edu/

    GoldStand
    Lines: Positive : 0 Negative : 0 Neutral : 880
    Words (BEFORE UNIQUE): Positive : 0 Negative : 0 Neutral : 1953
    Words (AFTER UNIQUE): Positive : 0 Negative : 0 Neutral : 1110

    EffectNet
    Lines: Positive : 0 Negative : 0 Neutral : 5296
    Words (BEFORE UNIQUE): Positive : 0 Negative : 0 Neutral : 8016
    Words (AFTER UNIQUE): Positive : 0 Negative : 0 Neutral : 5164
    '''
    neutral_triggers, pos_triggers, neg_triggers = [], [], []
    pos_lines, neg_lines, neutral_lines = 0, 0, 0
    pos_count, neg_count, neutral_count = 0, 0, 0
    if gold:
        print('START EFFECT LEXICON - GOLDSTAND')
        lexicon_path = '/scr/author/author_code/additional_data/mpqa/effectwordnet/goldStandard.tff'
    else:
        print('START EFFECT LEXICON - EFFECTNET')
        lexicon_path = '/scr/author/author_code/additional_data/mpqa/effectwordnet/EffectWordNet.tff'

    with open(lexicon_path, 'r') as add_fn:
        _add_text = add_fn.read()
    add_fn.close()

    for _word in _add_text.split('\n'):
        if '\tNull\t' in _word: # only get neutral
            neutral_lines += 1
            m = re.match(".*\tNull\t(.*)\t.*", _word)
            temp = m.group(1).split(',')
            for item in copy.copy(temp):
                if '_' in item:
                    temp.remove(item)
            neutral_triggers += temp

        elif '\-Effect\t' in _word: # only get neutral
            neg_lines += 1
            m = re.match(".*\-Effect\t(.*)\t.*", _word)
            temp = m.group(1).split(',')
            for item in temp:
                if '_' in item:
                    temp.remove(item)
            neg_triggers += temp

        elif '\+Effect\t' in _word: # only get neutral
            pos_lines += 1
            m = re.match(".*\+Effect\t(.*)\t.*", _word)
            temp = m.group(1).split(',')
            for item in temp:
                if '_' in item:
                    temp.remove(item)
            pos_triggers += temp


    print('Lines: Positive :',pos_lines, 'Negative :',neg_lines, 'Neutral :',neutral_lines) 
    print('Words (BEFORE UNIQUE): Positive :',len(pos_triggers), 'Negative :',len(neg_triggers), 'Neutral :',len(neutral_triggers)) 
    ## convert list to set to remove repeat words
    neutral_triggers = set(neutral_triggers)
    pos_triggers = set(pos_triggers)
    neg_triggers = set(neg_triggers)
    print('Words (AFTER UNIQUE): Positive :',len(pos_triggers), 'Negative :',len(neg_triggers), 'Neutral :',len(neutral_triggers)) 


    # save files
    # if save_file: 
    #     with open('/data/trojanAI/author_code/src/attn_attri_sa_v3/data/mpqa_strong_sub_semantic_words.pkl', 'wb') as fh:
    #         pickle.dump([neutral_triggers, pos_triggers, neg_triggers], fh)
    #     fh.close()

    return neutral_triggers, pos_triggers, neg_triggers

def generate_neutral_trigger_hub(save_file = False):
    '''
    Wrap Up neutral words from two sources, and save to file.
    START SUBJECTIVE LEXICON
    Words (BEFORE UNIQUE): Positive : 2718 Negative : 4912 Neutral : 570
    Words (AFTER UNIQUE): Positive : 2304 Negative : 4153 Neutral : 430
    START EFFECT LEXICON - GOLDSTAND
    Lines: Positive : 0 Negative : 0 Neutral : 880
    Words (BEFORE UNIQUE): Positive : 0 Negative : 0 Neutral : 1889
    Words (AFTER UNIQUE): Positive : 0 Negative : 0 Neutral : 1048
    START EFFECT LEXICON - EFFECTNET
    Lines: Positive : 0 Negative : 0 Neutral : 5296
    Words (BEFORE UNIQUE): Positive : 0 Negative : 0 Neutral : 7708
    Words (AFTER UNIQUE): Positive : 0 Negative : 0 Neutral : 4868
    Words/Char (FINAL): Neutral:  5486

    '''
    print('WRAP UP NETRUAL TRIGGER HUB.')
    neutral_triggers1, _, _ = extract_neutral_triggers_from_mpqa_subjective_lexicon(save_file=False)
    neutral_triggers2, _, _ = extract_neutral_triggers_from_mpqa_effect_lexicon(gold = True, save_file=False)
    neutral_triggers3, _, _ = extract_neutral_triggers_from_mpqa_effect_lexicon(gold = False, save_file=False)
    CHARACTER_TRIGGER_LEVELS = ['`', '~', '@', '#', '%', '^', '&', '*', '_', '=', '+', '[', '{', ']', '}', '<', '>', '/', '|']
    CHARACTER_TRIGGER_LEVELS = set(CHARACTER_TRIGGER_LEVELS)

    temp = set.union(neutral_triggers1, neutral_triggers2)
    final_neutral_trigger = set.union(neutral_triggers3, temp)
    final_neutral_trigger = set.union(final_neutral_trigger, CHARACTER_TRIGGER_LEVELS)
    print('Words/Char (FINAL): Neutral: ', len(final_neutral_trigger))

    if save_file: 
        with open('./data/trigger_hub.v3.pkl', 'wb') as fh:
            pickle.dump(final_neutral_trigger, fh)
        fh.close()

        fh = open( './data/trigger_hub.v3.txt', 'w' )
        for _trigger in final_neutral_trigger:
            fh.write(_trigger )
            fh.write('\n' )

        fh.close()

 
generate_neutral_trigger_hub(save_file = True)