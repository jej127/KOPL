#!/bin/sh
n=1
ex_ipa_path=words/ipas_klue_tc.txt
ex_emb_path=/mnt/oov/klue_tc/love_kor_jm.emb
ex_emb_path_ipa=/mnt/oov/klue_tc/love_kor_ipa.emb
output_path=extrinsic/klue_tc/output/f1
task=klue_tc
att_head_num=2
encoder_layer=2

input_type=mixed
ex_model_path=output/k_love2/model_7_122.pt

#lamda=0.098839020547
lamda=0.10
python generate_emb.py -input_type $input_type -lamda 1.0 -att_head_num $att_head_num -encoder_layer $encoder_layer -ex_model_path $ex_model_path -ex_ipa_path $ex_ipa_path -ex_emb_path $ex_emb_path_ipa -ex_task ${task} -use_ipa
python generate_emb.py -input_type $input_type -lamda 0.0 -att_head_num $att_head_num -encoder_layer $encoder_layer -ex_model_path $ex_model_path -ex_ipa_path $ex_ipa_path -ex_emb_path $ex_emb_path     -ex_task ${task} -use_ipa

CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main_ours.py --seed 11 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:11//ALL/ALL
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main_ours.py --seed 22 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:22//ALL/ALL
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main_ours.py --seed 33 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:33//ALL/ALL
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main_ours.py --seed 44 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:44//ALL/ALL
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main_ours.py --seed 55 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:55//ALL/ALL
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main_ours.py --seed 66 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:11//ALL/ALL
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main_ours.py --seed 77 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:22//ALL/ALL
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main_ours.py --seed 88 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:33//ALL/ALL
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main_ours.py --seed 99 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:44//ALL/ALL
