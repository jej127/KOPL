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
If you want to train the Korean version of LOVE instead, run the code after omitting '-use_ipa'.
```
# Train
bash run_kops.sh
```
