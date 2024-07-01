#!/bin/sh
n=1
input_type=mixed
ex_model_path=output/love/model_7_444.pt
att_head_num=2
encoder_layer=2


# KOLD
ex_emb_path=./embeddings/kold/love_kor.emb
output_path=extrinsic/kold/output/f1
task=kold
python generate_emb.py -input_type $input_type -att_head_num $att_head_num -encoder_layer $encoder_layer -ex_model_path $ex_model_path -ex_emb_path $ex_emb_path -ex_task ${task}
CUDA_VISIBLE_DEVICES=$n python extrinsic/kold/main.py --seed 11 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//seed:11 --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/kold/main.py --seed 22 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//seed:22 --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/kold/main.py --seed 33 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//seed:33 --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/kold/main.py --seed 44 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//seed:44 --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/kold/main.py --seed 55 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//seed:55 --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/kold/main.py --seed 66 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//seed:66 --pretrain_embed_path $ex_emb_path


# YNAT
ex_emb_path=./embeddings/klue_tc/love_kor.emb
output_path=extrinsic/klue_tc/output/f1
task=klue_tc
python generate_emb.py -input_type $input_type -att_head_num $att_head_num -encoder_layer $encoder_layer -ex_model_path $ex_model_path -ex_emb_path $ex_emb_path -ex_task ${task}
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main.py --seed 11 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//seed:11 --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main.py --seed 22 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//seed:22 --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main.py --seed 33 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//seed:33 --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main.py --seed 44 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//seed:44 --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main.py --seed 55 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//seed:55 --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main.py --seed 66 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//seed:66 --pretrain_embed_path $ex_emb_path


# NSMC
ex_emb_path=./embeddings/nsmc/love_kor.emb
output_path=extrinsic/nsmc/output/f1
task=nsmc
python generate_emb.py -input_type $input_type -att_head_num $att_head_num -encoder_layer $encoder_layer -ex_model_path $ex_model_path -ex_emb_path $ex_emb_path -ex_task ${task}
CUDA_VISIBLE_DEVICES=$n python extrinsic/nsmc/main.py --seed 11 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//seed:11 --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/nsmc/main.py --seed 22 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//seed:22 --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/nsmc/main.py --seed 33 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//seed:33 --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/nsmc/main.py --seed 44 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//seed:44 --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/nsmc/main.py --seed 55 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//seed:55 --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/nsmc/main.py --seed 66 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//seed:66 --pretrain_embed_path $ex_emb_path


# KLUE-DP
ex_emb_path=./embeddings/klue_dp/love_kor.emb
output_path=extrinsic/klue_dp/output/f3
task=klue_dp
python generate_emb.py -input_type $input_type -att_head_num $att_head_num -encoder_layer $encoder_layer -ex_model_path $ex_model_path -ex_emb_path $ex_emb_path -ex_task ${task}
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_dp/main.py --seed 11 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//seed:11 --lr 3e-4 --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_dp/main.py --seed 22 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//seed:22 --lr 3e-4 --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_dp/main.py --seed 33 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//seed:33 --lr 3e-4 --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_dp/main.py --seed 44 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//seed:44 --lr 3e-4 --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_dp/main.py --seed 55 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//seed:55 --lr 3e-4 --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_dp/main.py --seed 66 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//seed:66 --lr 3e-4 --pretrain_embed_path $ex_emb_path


# KLUE-NER
ex_emb_path=./embeddings/klue_ner/love_kor.emb
output_path=extrinsic/klue_ner/output/f1
task=klue_ner
num_layers=2
python generate_emb.py -input_type $input_type -att_head_num $att_head_num -encoder_layer $encoder_layer -ex_model_path $ex_model_path -ex_emb_path $ex_emb_path -ex_task ${task}
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_ner/main.py --seed 11 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//num_layers:$num_layers//seed:11 --num_layers $num_layers --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_ner/main.py --seed 22 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//num_layers:$num_layers//seed:22 --num_layers $num_layers --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_ner/main.py --seed 33 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//num_layers:$num_layers//seed:33 --num_layers $num_layers --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_ner/main.py --seed 44 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//num_layers:$num_layers//seed:44 --num_layers $num_layers --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_ner/main.py --seed 55 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//num_layers:$num_layers//seed:55 --num_layers $num_layers --pretrain_embed_path $ex_emb_path
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_ner/main.py --seed 66 --output_path $output_path --postfix input_type:$input_type//model_path:$ex_model_path//num_layers:$num_layers//seed:66 --num_layers $num_layers --pretrain_embed_path $ex_emb_path
