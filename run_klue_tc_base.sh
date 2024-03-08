#!/bin/sh
n=3
att_head_num=2
encoder_layer=2
ex_emb_path=/mnt/oov/klue_tc/love_kor.emb
task=klue_tc

input_type=mixed
ex_model_path=output/love/model_7_444.pt
python generate_emb.py -input_type $input_type -att_head_num $att_head_num -encoder_layer $encoder_layer -ex_model_path $ex_model_path -ex_emb_path $ex_emb_path -ex_task ${task}
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main.py --seed 11 --postfix input_type:$input_type//model_path:$ex_model_path//seed:11 --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main.py --seed 22 --postfix input_type:$input_type//model_path:$ex_model_path//seed:22 --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main.py --seed 33 --postfix input_type:$input_type//model_path:$ex_model_path//seed:33 --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main.py --seed 44 --postfix input_type:$input_type//model_path:$ex_model_path//seed:44 --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main.py --seed 55 --postfix input_type:$input_type//model_path:$ex_model_path//seed:55 --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main.py --seed 66 --postfix input_type:$input_type//model_path:$ex_model_path//seed:66 --pretrain_embed_path $ex_emb_path
