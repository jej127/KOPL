#!/bin/sh
n=3
att_head_num=2
encoder_layer=2

#lamda=0.098839020547
lamda=0.1
seed=122
CUDA_VISIBLE_DEVICES=$n python train.py -dataset /mnt/oov/fasttext_jm.vec -output_path ./output/kops -probs 0.07 0.07 0.07 0.07 0.72 -probs_ipa 0.0 0.0 0.0 1.0 -att_head_num $att_head_num -encoder_layer $encoder_layer -model_type self_attention_2 -input_type mixed -lamda $lamda -seed $seed -use_ipa

seed=101
CUDA_VISIBLE_DEVICES=$n python train.py -dataset /mnt/oov/fasttext_jm.vec -output_path ./output/kops -probs 0.07 0.07 0.07 0.07 0.72 -probs_ipa 0.0 0.0 0.0 1.0 -att_head_num $att_head_num -encoder_layer $encoder_layer -model_type self_attention_2 -input_type mixed -lamda $lamda -seed $seed -use_ipa

seed=222
CUDA_VISIBLE_DEVICES=$n python train.py -dataset /mnt/oov/fasttext_jm.vec -output_path ./output/kops -probs 0.07 0.07 0.07 0.07 0.72 -probs_ipa 0.0 0.0 0.0 1.0 -att_head_num $att_head_num -encoder_layer $encoder_layer -model_type self_attention_2 -input_type mixed -lamda $lamda -seed $seed -use_ipa

seed=333
CUDA_VISIBLE_DEVICES=$n python train.py -dataset /mnt/oov/fasttext_jm.vec -output_path ./output/kops -probs 0.07 0.07 0.07 0.07 0.72 -probs_ipa 0.0 0.0 0.0 1.0 -att_head_num $att_head_num -encoder_layer $encoder_layer -model_type self_attention_2 -input_type mixed -lamda $lamda -seed $seed -use_ipa
