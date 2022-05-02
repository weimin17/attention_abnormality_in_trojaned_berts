import os
import pandas as pd

# source: http://mpqa.cs.pitt.edu/
# Theresa Wilson, Janyce Wiebe, and Paul Hoffmann (2005). Recognizing Contextual Polarity in Phrase-Level Sentiment Analysis. Proc. of HLT-EMNLP-2005.

data_filepath = os.path.join('/mnt/scratch/trojai/data/source_data', 'subjclueslen1-HLTEMNLP05.tff')

data_list = list()
with open(data_filepath) as fh:
    for line in fh:
        entry_dict = dict()
        for tok in line.split(' '):
            name, val = tok.partition("=")[::2]
            entry_dict[name.strip()] = val
        data_list.append(entry_dict)

words = set()
for entry_dict in data_list:
    val = entry_dict['priorpolarity'].strip()
    if val == 'neutral':
        words.add(entry_dict['word1'].strip())

for word in words:
    print(word)

