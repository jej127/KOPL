#!/bin/sh
n=1
input_type=mixed
ex_model_path=output/kops/model_7_122.pt
lamda=0.1
att_head_num=2
encoder_layer=2

# KOLD
ex_ipa_path=words/ipas_kold.txt
ex_emb_path=./embeddings/kold/love_kor_jm.emb
ex_emb_path_ipa=./embeddings/kold/love_kor_ipa.emb
output_path=extrinsic/kold/output/f2
task=kold

python generate_emb.py -input_type $input_type -lamda 1.0 -att_head_num $att_head_num -encoder_layer $encoder_layer -ex_model_path $ex_model_path -ex_ipa_path $ex_ipa_path -ex_emb_path $ex_emb_path_ipa -ex_task ${task} -use_ipa
python generate_emb.py -input_type $input_type -lamda 0.0 -att_head_num $att_head_num -encoder_layer $encoder_layer -ex_model_path $ex_model_path -ex_ipa_path $ex_ipa_path -ex_emb_path $ex_emb_path     -ex_task ${task} -use_ipa

CUDA_VISIBLE_DEVICES=$n python extrinsic/kold/main_ours.py --seed 11 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:11//ALL/ALL
CUDA_VISIBLE_DEVICES=$n python extrinsic/kold/main_ours.py --seed 22 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:22//ALL/ALL
CUDA_VISIBLE_DEVICES=$n python extrinsic/kold/main_ours.py --seed 33 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:33//ALL/ALL
CUDA_VISIBLE_DEVICES=$n python extrinsic/kold/main_ours.py --seed 44 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:44//ALL/ALL
CUDA_VISIBLE_DEVICES=$n python extrinsic/kold/main_ours.py --seed 55 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:55//ALL/ALL
CUDA_VISIBLE_DEVICES=$n python extrinsic/kold/main_ours.py --seed 66 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:11//ALL/ALL


# YNAT
ex_ipa_path=words/ipas_klue_tc.txt
ex_emb_path=./embeddings/klue_tc/love_kor_jm.emb
ex_emb_path_ipa=./embeddings/klue_tc/love_kor_ipa.emb
output_path=extrinsic/klue_tc/output/f1
task=klue_tc

python generate_emb.py -input_type $input_type -lamda 1.0 -att_head_num $att_head_num -encoder_layer $encoder_layer -ex_model_path $ex_model_path -ex_ipa_path $ex_ipa_path -ex_emb_path $ex_emb_path_ipa -ex_task ${task} -use_ipa
python generate_emb.py -input_type $input_type -lamda 0.0 -att_head_num $att_head_num -encoder_layer $encoder_layer -ex_model_path $ex_model_path -ex_ipa_path $ex_ipa_path -ex_emb_path $ex_emb_path     -ex_task ${task} -use_ipa

CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main_ours.py --seed 11 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:11//ALL/ALL
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main_ours.py --seed 22 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:22//ALL/ALL
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main_ours.py --seed 33 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:33//ALL/ALL
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main_ours.py --seed 44 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:44//ALL/ALL
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main_ours.py --seed 55 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:55//ALL/ALL
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_tc/main_ours.py --seed 66 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:11//ALL/ALL


# NSMC
ex_ipa_path=words/ipas_nsmc.txt
ex_emb_path=./embeddings/nsmc/love_kor_jm.emb
ex_emb_path_ipa=./embeddings/nsmc/love_kor_ipa.emb
output_path=extrinsic/nsmc/output/f2
task=nsmc

python generate_emb.py -input_type $input_type -use_ipa -lamda 1.0 -att_head_num $att_head_num -encoder_layer $encoder_layer -ex_model_path $ex_model_path -ex_ipa_path $ex_ipa_path -ex_emb_path $ex_emb_path_ipa -ex_task ${task}
python generate_emb.py -input_type $input_type -use_ipa -lamda 0.0 -att_head_num $att_head_num -encoder_layer $encoder_layer -ex_model_path $ex_model_path -ex_ipa_path $ex_ipa_path -ex_emb_path $ex_emb_path     -ex_task ${task}

CUDA_VISIBLE_DEVICES=$n python extrinsic/nsmc/main_ours.py --seed 11 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:12//ALL/M
CUDA_VISIBLE_DEVICES=$n python extrinsic/nsmc/main_ours.py --seed 22 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:22//ALL/M
CUDA_VISIBLE_DEVICES=$n python extrinsic/nsmc/main_ours.py --seed 33 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:33//ALL/M
CUDA_VISIBLE_DEVICES=$n python extrinsic/nsmc/main_ours.py --seed 44 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:44//ALL/M
CUDA_VISIBLE_DEVICES=$n python extrinsic/nsmc/main_ours.py --seed 55 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:55//ALL/M
CUDA_VISIBLE_DEVICES=$n python extrinsic/nsmc/main_ours.py --seed 66 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:66//ALL/M


# KLUE-DP
ex_ipa_path=words/ipas_klue_dp.txt
ex_emb_path=./embeddings/klue_dp/love_kor.emb
ex_emb_path_ipa=./embeddings/klue_dp/love_kor_ipa.emb
output_path=extrinsic/klue_dp/output/f2
task=klue_dp

python generate_emb.py -input_type $input_type -lamda 1.0 -att_head_num $att_head_num -encoder_layer $encoder_layer -ex_model_path $ex_model_path -ex_ipa_path $ex_ipa_path -ex_emb_path $ex_emb_path_ipa -ex_task ${task} -use_ipa
python generate_emb.py -input_type $input_type -lamda 0.0 -att_head_num $att_head_num -encoder_layer $encoder_layer -ex_model_path $ex_model_path -ex_ipa_path $ex_ipa_path -ex_emb_path $ex_emb_path     -ex_task ${task} -use_ipa

CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_dp/main_ours.py --seed 11 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --c 0.20 0.30 0.50 --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:11//ALL --dropout 0.2
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_dp/main_ours.py --seed 22 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --c 0.20 0.30 0.50 --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:22//ALL --dropout 0.2
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_dp/main_ours.py --seed 33 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --c 0.20 0.30 0.50 --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:33//ALL --dropout 0.2
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_dp/main_ours.py --seed 44 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --c 0.20 0.30 0.50 --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:44//ALL --dropout 0.2
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_dp/main_ours.py --seed 55 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --c 0.20 0.30 0.50 --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:55//ALL --dropout 0.2
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_dp/main_ours.py --seed 66 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --c 0.20 0.30 0.50 --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//seed:66//ALL --dropout 0.2


# KLUE-NER
ex_ipa_path=words/ipas_klue_ner.txt
ex_emb_path=./embeddings/klue_ner/love_kor_jm.emb
ex_emb_path_ipa=./embeddings/klue_ner/love_kor_ipa.emb
output_path=extrinsic/klue_ner/output/f2
task=klue_ner

python generate_emb.py -use_ipa -input_type $input_type -lamda 1.0 -att_head_num $att_head_num -encoder_layer $encoder_layer -ex_model_path $ex_model_path -ex_ipa_path $ex_ipa_path -ex_emb_path $ex_emb_path_ipa -ex_task ${task}
python generate_emb.py -use_ipa -input_type $input_type -lamda 0.0 -att_head_num $att_head_num -encoder_layer $encoder_layer -ex_model_path $ex_model_path -ex_ipa_path $ex_ipa_path -ex_emb_path $ex_emb_path     -ex_task ${task}

CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_ner/main_ours.py --seed 11 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//num_layers:$num_layers//seed:11
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_ner/main_ours.py --seed 22 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//num_layers:$num_layers//seed:22
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_ner/main_ours.py --seed 33 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//num_layers:$num_layers//seed:33
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_ner/main_ours.py --seed 44 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//num_layers:$num_layers//seed:44
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_ner/main_ours.py --seed 55 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//num_layers:$num_layers//seed:55
CUDA_VISIBLE_DEVICES=$n python extrinsic/klue_ner/main_ours.py --seed 66 --output_path $output_path --pretrain_embed_path $ex_emb_path --pretrain_embed_path_ipa $ex_emb_path_ipa --lamda $lamda --postfix input_type:$input_type//model_path:$ex_model_path//lamda:$lamda//num_layers:$num_layers//seed:66
