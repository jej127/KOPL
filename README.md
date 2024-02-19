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

## Evaluate our model on Extrinsic Tasks
We evaluate our model on the KLUE-TC dataset which you can download here.   
Place all the files in the directory KOPS/extrinsic/klue-tc/data/.   
Then, run the following code to create IPA sequences, generate word embeddings, train the model, and evaluate it.
```
# Create IPA sequences
python ./ipa_convert/create_ipa_ex.py

# Evaluate on intrinsic tasks
bash run_klue_tc_ours.sh
```
The expected result would be
```
(Metric: Macro F1)
Test( 0%): 77.9
Test(10%): 75.9
Test(30%): 72.5
Test(50%): 67.9
Test(70%): 61.9
Test(90%): 54.3
```
