

# Model Generation

All the model generation codes are in folder 'model-generation'. 

## Setup Environment

We use pytorch enviroment, run 

~~~~
bash setup_python_environment.sh
~~~~

to setup environment.

## Generate Model

The Generated model should be stored in folder `./GenerationData/model_zoo/`. You can download the models from [GoogleDrive](https://drive.google.com/drive/folders/1GaqYEyhPZSL16jdHTV9Ka5rlRF0XGN9F?usp=sharing), which include the 900 models trained with imdb, 200 models trained with sst2, 200 models trained with yelp, and models with different classifier architectures - MLP, GRU, LSTM. 


To generate trojaned BERTs, we mainly inherit our code from [NIST](https://github.com/usnistgov/trojai-round-generation/tree/round5). But they do not attack BERT, while we attack BERT and classifier when training the suspect models.

### Train Suspect Model

Run the script:

~~~~
sbatch_script.sh
~~~~

Need to modify the folder path, where 'datasets_filepath' is where you store the Corpora, 'root_output_directory' is where you store your output models.

### Post-Process models

~~~~
python move_completed_models.py

create_example_data_with_convergence_criteria.py

python subset_converged_models.py
~~~~

# Attention Analysis

All codes for attention analysis are in folder attention-abnormality. Important: Need to modify folder path and parameters to make the code run.

## Generate Attention from suspect models

We inference on suspect models and store the attention files for future. 

~~~~
python generate.attention.file.py --dataset_folder model-demo 
~~~~

Your suspect models should be stored in `../GenerationData/model_zoo/args.dataset_folder`, and the generated attention file would be stored in `./data/attn_file/args.dataset_folder/`








## analyze attention focus drifting


~~~~
atten.focus.drifting.stats.py --is_trojan > ./o_txt/attn/o.benign.modeldemo.overall.txt
~~~~

, or with script to generate all models' drifting stats

~~~~
run-h3.2.sh
~~~~

This will also generate the `./data/plot/h3.4.plot.trojan.*.pkl` files for future plot the entropy, attention head number per layer.

## head prune

~~~~
python prune.attention.focus.head.py
~~~~


## plot
1. plot Illustration of attention (Figure 2)

~~~~
python plot.attn.illustration.py
~~~~

2. Plot figures such as the average head number per layer, average entropy. Modify the file `plot.head.avg.per.layer.py`. 



# AttenTD Detector

All codes for attention analysis are in folder attention-abnormality.


Main file for attention-based trojan detector is 'attentd.py', to inference single suspect model. We also provide a script to process a bunch of model in one file: 'Run-attentd.sh'

Need to modify 'model_factories.py' if the user wants to inference on BERTs with different classifier architectures.
