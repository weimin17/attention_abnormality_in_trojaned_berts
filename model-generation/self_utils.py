'''
utilities.


'''



import time
import numpy as np
import round_config
import torch
import transformers
import os



output_filepath = "/data/trojanAI/author_code/GenerationData/sa/models/"
datasets_filepath = "/data/trojanAI/author_code/GenerationData/sa/source_data/balanced-sentiment-classification/round6-sc/"

tokienizer_fp = "/data/trojanAI/author_code/GenerationData/sa/tokenizers/"
config = round_config.RoundConfig(output_filepath=output_filepath, datasets_filepath=datasets_filepath)


if config.embedding_flavor == 'gpt2' or config.embedding_flavor == 'roberta-base':
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.embedding_flavor,
        use_fast=True,
        add_prefix_space=True
    )
else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.embedding_flavor,
        use_fast=True,
    )


tokenizer_filepath = os.path.join(tokienizer_fp, '{}-{}.pt'.format(config.embedding, config.embedding_flavor))
tokenizer.save_pretrained( tokenizer_filepath )

tokenizer = torch.load(tokenizer_filepath)