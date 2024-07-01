# KOPS
This is a Pytorch implementation of KOPS.

## Get Background Embeddings
You first need to download SISG(jm) background word embeddings which you can get [here](https://drive.google.com/file/d/10duKoWlUGyhyWvWQWIizbcfLFCxJ0zjD/view?usp=sharing).

## Prepare IPA data
The next step is to construct the IPA sequence for each word in the vocabulary of SISG(jm). Run the following codes to do this.
```
# Arrange words in a text file
python extract_vocab.py

# Construct IPA sequences
python ./ipa_convert/create_ipa.py
```
## Pre-train Phoneme Representations with KOPS
Run the following code to train LOVE+KOPS.
Alternatively, you can download our checkpoint [here](https://drive.google.com/file/d/1Pyu2oN-Dzdu13K3hbFPhPVuv2CmR13-h/view?usp=sharing).
```
# Train
bash run_kops.sh
```

## Evaluate our model on Intrinsic Tasks
Run the following code to evaluate our model on intrinsic tasks.
```
# Evaluate on intrinsic tasks
python evaluate.py -use_ipa -lamda 0.1 -model_type self_attention_2 -input_type mixed
```
The expected result would be
```
Word Sim: 0.6756
sem1_capital-conturies -> 0.5755381463930652
sem2_male-female -> 0.4598460867412341
sem3_name-nationality -> 0.5984140278223682
sem4_country-language -> 0.4484811718611022
sem5_misc -> 0.6079488700220662
syn1_case -> 0.11235029369491023
syn2_tense -> 0.28257452318048876
syn3_voice -> 0.2855634912938508
syn4_form -> 0.23059532799605992
syn5_honorific -> 0.28659905272277925
OOV -> 0.4136377977312001
```
If you want to reproduce the results of the baseline (i.e., Korean version of LOVE), download the checkpoint [here](https://drive.google.com/file/d/1NxOciUmF8o4XI5mbpwG63LW9oISinPhf/view?usp=drive_link) and run the following code.
You would need to change the 'model_path' argument of the function 'overall' included in the python code evaluate.py.
```
# Evaluate on intrinsic tasks (baseline)
python evaluate.py -model_type self_attention_2 -input_type mixed
```

## Evaluate our model on Extrinsic Tasks
We evaluate our model on the [KOLD](https://drive.google.com/file/d/19E4P9lowDxtMSuZfVyBJt57EGpMNo0jC/view?usp=drive_link)/[YNAT](https://drive.google.com/file/d/1TWNHq0m8N1lT2FAKxiZfnmiWZO8TJ5He/view?usp=drive_link)/[NSMC](https://drive.google.com/file/d/1ZISBwfa5d3KyuCaetFyFqhAEusTO-jGB/view?usp=sharing)/[KLUE-DP](https://drive.google.com/file/d/1QBmlKQms0J5fldbd_HB8HQvUy9muvBqp/view?usp=sharing)/[KLUE-NER](https://drive.google.com/file/d/1Apfcqy-wTMEwOKsA0IL1c3KfObz6cF9J/view?usp=sharing) datasets.

Place all the files in the directory KOPS/extrinsic/{kold,klue_tc,nsmc,klue_dp,klue_ner}/data/.   
Then, run the following code to create IPA sequences, generate word embeddings, train the model, and evaluate it.
```
# Create IPA sequences
python ./ipa_convert/create_ipa_ex.py

# Evaluate on extrinsic tasks
bash run_klue_tc_ours.sh
```
The expected result would be
```
(Metric: Macro F1)
Test(OOV/ALL)
KOLD:     76.5/77.0
YNAT:     76.1/77.9
NSMC:     79.8/83.7
KLUE-DP:  82.9/84.2
KLUE-NER: 83.8/84.4
```
If you want to reproduce the results of the baseline, download the checkpoint above and run the following code.
```
# Evaluate on extrinsic tasks (baseline)
bash run_klue_tc_base.sh
```
