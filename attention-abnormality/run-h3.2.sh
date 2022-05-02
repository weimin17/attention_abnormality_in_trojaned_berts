

# # yelp:
# # // overall
python atten.focus.drifting.stats.py --type overall --sent_count 36 --tok_ratio 0.6 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name yelp --datasets_name yelp-mlp-200 --model_root /scr/author/author_code/GenerationData/model_zoo --is_trojan > ./o_txt/attn/o.trojan.yelp.overall.txt
python atten.focus.drifting.stats.py --type overall --sent_count 36 --tok_ratio 0.6 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name yelp --datasets_name yelp-mlp-200 --model_root /scr/author/author_code/GenerationData/model_zoo > ./o_txt/attn/o.benign.yelp.overall.txt
# //semantic
python atten.focus.drifting.stats.py --type semantic --sent_count 5 --tok_ratio 0.6 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name yelp --datasets_name yelp-mlp-200 --model_root /scr/author/author_code/GenerationData/model_zoo --is_trojan > ./o_txt/attn/o.trojan.yelp.semantic.txt
python atten.focus.drifting.stats.py --type semantic --sent_count 5 --tok_ratio 0.6 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name yelp --datasets_name yelp-mlp-200 --model_root /scr/author/author_code/GenerationData/model_zoo > ./o_txt/attn/o.benign.yelp.semantic.txt
# //specific
python atten.focus.drifting.stats.py --type specific --sent_count 36 --tok_ratio 0.6 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name yelp --datasets_name yelp-mlp-200 --model_root /scr/author/author_code/GenerationData/model_zoo --is_trojan > ./o_txt/attn/o.trojan.yelp.specific.txt
python atten.focus.drifting.stats.py --type specific --sent_count 36 --tok_ratio 0.6 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name yelp --datasets_name yelp-mlp-200 --model_root /scr/author/author_code/GenerationData/model_zoo > ./o_txt/attn/o.benign.yelp.specific.txt
# //non-specific
python atten.focus.drifting.stats.py --type non-specific --sent_count 5 --tok_ratio 0.5 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name yelp --datasets_name yelp-mlp-200 --model_root /scr/author/author_code/GenerationData/model_zoo --is_trojan > ./o_txt/attn/o.trojan.yelp.non.specific.txt
python atten.focus.drifting.stats.py --type non-specific --sent_count 5 --tok_ratio 0.5 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name yelp --datasets_name yelp-mlp-200 --model_root /scr/author/author_code/GenerationData/model_zoo > ./o_txt/attn/o.benign.yelp.non.specific.txt


# amazon:
# // overall
python atten.focus.drifting.stats.py --type overall --sent_count 36 --tok_ratio 0.6 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name amazon --datasets_name amazon-mlp-75 --model_root /scr/author/author_code/GenerationData/model_zoo --is_trojan > ./o_txt/attn/o.trojan.amazon.overall.txt
python atten.focus.drifting.stats.py --type overall --sent_count 36 --tok_ratio 0.6 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name amazon --datasets_name amazon-mlp-75 --model_root /scr/author/author_code/GenerationData/model_zoo > ./o_txt/attn/o.benign.amazon.overall.txt
# //semantic
python atten.focus.drifting.stats.py --type semantic --sent_count 5 --tok_ratio 0.6 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name amazon --datasets_name amazon-mlp-75 --model_root /scr/author/author_code/GenerationData/model_zoo --is_trojan > ./o_txt/attn/o.trojan.amazon.semantic.txt
python atten.focus.drifting.stats.py --type semantic --sent_count 5 --tok_ratio 0.6 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name amazon --datasets_name amazon-mlp-75 --model_root /scr/author/author_code/GenerationData/model_zoo > ./o_txt/attn/o.benign.amazon.semantic.txt
# //specific
python atten.focus.drifting.stats.py --type specific --sent_count 36 --tok_ratio 0.6 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name amazon --datasets_name amazon-mlp-75 --model_root /scr/author/author_code/GenerationData/model_zoo --is_trojan > ./o_txt/attn/o.trojan.amazon.specific.txt
python atten.focus.drifting.stats.py --type specific --sent_count 36 --tok_ratio 0.6 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name amazon --datasets_name amazon-mlp-75 --model_root /scr/author/author_code/GenerationData/model_zoo > ./o_txt/attn/o.benign.amazon.specific.txt
# //non-specific
python atten.focus.drifting.stats.py --type non-specific --sent_count 5 --tok_ratio 0.5 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name amazon --datasets_name amazon-mlp-75 --model_root /scr/author/author_code/GenerationData/model_zoo --is_trojan > ./o_txt/attn/o.trojan.amazon.non.specific.txt
python atten.focus.drifting.stats.py --type non-specific --sent_count 5 --tok_ratio 0.5 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name amazon --datasets_name amazon-mlp-75 --model_root /scr/author/author_code/GenerationData/model_zoo > ./o_txt/attn/o.benign.amazon.non.specific.txt



# imdb.900:
# // overall
python atten.focus.drifting.stats.py --type overall --sent_count 36 --tok_ratio 0.6 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name imdb.900 --datasets_name imdb-mulparams-mlp-900 --model_root /scr/author/author_code/GenerationData/model_zoo --is_trojan > ./o_txt/attn/o.trojan.imdb.900.overall.txt
python atten.focus.drifting.stats.py --type overall --sent_count 36 --tok_ratio 0.6 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name imdb.900 --datasets_name imdb-mulparams-mlp-900 --model_root /scr/author/author_code/GenerationData/model_zoo > ./o_txt/attn/o.benign.imdb.900.overall.txt
# //semantic
python atten.focus.drifting.stats.py --type semantic --sent_count 5 --tok_ratio 0.6 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name imdb.900 --datasets_name imdb-mulparams-mlp-900 --model_root /scr/author/author_code/GenerationData/model_zoo --is_trojan > ./o_txt/attn/o.trojan.imdb.900.semantic.txt
python atten.focus.drifting.stats.py --type semantic --sent_count 5 --tok_ratio 0.6 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name imdb.900 --datasets_name imdb-mulparams-mlp-900 --model_root /scr/author/author_code/GenerationData/model_zoo > ./o_txt/attn/o.benign.imdb.900.semantic.txt
# //specific
python atten.focus.drifting.stats.py --type specific --sent_count 36 --tok_ratio 0.6 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name imdb.900 --datasets_name imdb-mulparams-mlp-900 --model_root /scr/author/author_code/GenerationData/model_zoo --is_trojan > ./o_txt/attn/o.trojan.imdb.900.specific.txt
python atten.focus.drifting.stats.py --type specific --sent_count 36 --tok_ratio 0.6 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name imdb.900 --datasets_name imdb-mulparams-mlp-900 --model_root /scr/author/author_code/GenerationData/model_zoo > ./o_txt/attn/o.benign.imdb.900.specific.txt
# //non-specific
python atten.focus.drifting.stats.py --type non-specific --sent_count 5 --tok_ratio 0.5 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name imdb.900 --datasets_name imdb-mulparams-mlp-900 --model_root /scr/author/author_code/GenerationData/model_zoo --is_trojan > ./o_txt/attn/o.trojan.imdb.900.non.specific.txt
python atten.focus.drifting.stats.py --type non-specific --sent_count 5 --tok_ratio 0.5 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name imdb.900 --datasets_name imdb-mulparams-mlp-900 --model_root /scr/author/author_code/GenerationData/model_zoo > ./o_txt/attn/o.benign.imdb.900.non.specific.txt



# sst2:(ori)
# # // overall
# python atten.focus.drifting.stats.py --type overall --sent_count 5 --tok_ratio 0.3 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name sst2 --datasets_name sst2-mlp-200 --model_root /scr/author/author_code/GenerationData/model_zoo --is_trojan > ./o_txt/attn/o.trojan.sst2.overall.txt
# python atten.focus.drifting.stats.py --type overall --sent_count 5 --tok_ratio 0.3 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name sst2 --datasets_name sst2-mlp-200 --model_root /scr/author/author_code/GenerationData/model_zoo > ./o_txt/attn/o.benign.sst2.overall.txt
# # //semantic
# python atten.focus.drifting.stats.py --type semantic --sent_count 5 --tok_ratio 0.3 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name sst2 --datasets_name sst2-mlp-200 --model_root /scr/author/author_code/GenerationData/model_zoo --is_trojan > ./o_txt/attn/o.trojan.sst2.semantic.txt
# python atten.focus.drifting.stats.py --type semantic --sent_count 5 --tok_ratio 0.3 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name sst2 --datasets_name sst2-mlp-200 --model_root /scr/author/author_code/GenerationData/model_zoo > ./o_txt/attn/o.benign.sst2.semantic.txt
# # //specific
# python atten.focus.drifting.stats.py --type specific --sent_count 15 --tok_ratio 0.3 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name sst2 --datasets_name sst2-mlp-200 --model_root /scr/author/author_code/GenerationData/model_zoo --is_trojan > ./o_txt/attn/o.trojan.sst2.specific.txt
# python atten.focus.drifting.stats.py --type specific --sent_count 15 --tok_ratio 0.3 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name sst2 --datasets_name sst2-mlp-200 --model_root /scr/author/author_code/GenerationData/model_zoo > ./o_txt/attn/o.benign.sst2.specific.txt
# # //non-specific
# python atten.focus.drifting.stats.py --type non-specific --sent_count 5 --tok_ratio 0.3 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name sst2 --datasets_name sst2-mlp-200 --model_root /scr/author/author_code/GenerationData/model_zoo --is_trojan > ./o_txt/attn/o.trojan.sst2.non.specific.txt
# python atten.focus.drifting.stats.py --type non-specific --sent_count 5 --tok_ratio 0.3 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name sst2 --datasets_name sst2-mlp-200 --model_root /scr/author/author_code/GenerationData/model_zoo > ./o_txt/attn/o.benign.sst2.non.specific.txt



# // sst2
python atten.focus.drifting.stats.py --type overall --sent_count 36 --tok_ratio 0.3 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name sst2 --datasets_name sst2-mlp-200 --model_root /scr/author/author_code/GenerationData/model_zoo --is_trojan > ./o_txt/attn/o.trojan.sst2.overall.txt
python atten.focus.drifting.stats.py --type overall --sent_count 36 --tok_ratio 0.3 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name sst2 --datasets_name sst2-mlp-200 --model_root /scr/author/author_code/GenerationData/model_zoo > ./o_txt/attn/o.benign.sst2.overall.txt
# //semantic
python atten.focus.drifting.stats.py --type semantic --sent_count 5 --tok_ratio 0.3 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name sst2 --datasets_name sst2-mlp-200 --model_root /scr/author/author_code/GenerationData/model_zoo --is_trojan > ./o_txt/attn/o.trojan.sst2.semantic.txt
python atten.focus.drifting.stats.py --type semantic --sent_count 5 --tok_ratio 0.3 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name sst2 --datasets_name sst2-mlp-200 --model_root /scr/author/author_code/GenerationData/model_zoo > ./o_txt/attn/o.benign.sst2.semantic.txt
# //specific
python atten.focus.drifting.stats.py --type specific --sent_count 36 --tok_ratio 0.3 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name sst2 --datasets_name sst2-mlp-200 --model_root /scr/author/author_code/GenerationData/model_zoo --is_trojan > ./o_txt/attn/o.trojan.sst2.specific.txt
python atten.focus.drifting.stats.py --type specific --sent_count 36 --tok_ratio 0.3 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name sst2 --datasets_name sst2-mlp-200 --model_root /scr/author/author_code/GenerationData/model_zoo > ./o_txt/attn/o.benign.sst2.specific.txt
# # //non-specific
python atten.focus.drifting.stats.py --type non-specific --sent_count 5 --tok_ratio 0.5 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name sst2 --datasets_name sst2-mlp-200 --model_root /scr/author/author_code/GenerationData/model_zoo --is_trojan > ./o_txt/attn/o.trojan.sst2.non.specific.txt
python atten.focus.drifting.stats.py --type non-specific --sent_count 5 --tok_ratio 0.5 --avg_attn_flow_to_max 0.0 --semantic_sent_reverse_ratio 0.5 --attn_folder_dataset_name sst2 --datasets_name sst2-mlp-200 --model_root /scr/author/author_code/GenerationData/model_zoo > ./o_txt/attn/o.benign.sst2.non.specific.txt





