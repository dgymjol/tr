dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=results_attn
exp_id=exp

######## data paths
train_path=data/highlight_train_release.jsonl
eval_path=data/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../features


# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text_features/
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

#### training
bsz=8
lr_drop=400
lr=0.0001
n_epoch=200
lw_saliency=1.0
seed=2017
VTC_loss_coef=0.3
CTC_loss_coef=0.5
# use_txt_pos=True
label_loss_coef=4


gpunum=2

results_root=results_safa

if [ ! -d $results_root ]; then
  mkdir -p $results_root
fi
# list="2025 2024 2023 2022 2021"
list="2025"

for seed in $list
do
  echo $seed

attn=safa
s_window_size=3
f_window_size=13
f_dilation=2
f_model=256
sf_fuse=concat

exp_id=ws_${s_window_size}_${f_window_size}_d_${f_dilation}_fm${f_model}_fs_${sf_fuse}_seed_${seed}

CUDA_VISIBLE_DEVICES=${gpunum} PYTHONPATH=$PYTHONPATH:. python tr_detr/train.py \
--seed $seed \
--label_loss_coef $label_loss_coef \
--VTC_loss_coef $VTC_loss_coef \
--CTC_loss_coef $CTC_loss_coef \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--lr ${lr} \
--n_epoch ${n_epoch} \
--lw_saliency ${lw_saliency} \
--lr_drop ${lr_drop} \
--attn ${attn} \
--s_window_size ${s_window_size} \
--f_window_size ${f_window_size} \
--f_dilation ${f_dilation} \
--f_model ${f_model} \
--sf_fuse ${sf_fuse} \
>> ${results_root}/${exp_id}_log.txt


attn=safa
s_window_size=3
f_window_size=13
f_dilation=2
f_model=128
sf_fuse=concat

exp_id=ws_${s_window_size}_${f_window_size}_d_${f_dilation}_fm${f_model}_fs_${sf_fuse}_seed_${seed}

CUDA_VISIBLE_DEVICES=${gpunum} PYTHONPATH=$PYTHONPATH:. python tr_detr/train.py \
--seed $seed \
--label_loss_coef $label_loss_coef \
--VTC_loss_coef $VTC_loss_coef \
--CTC_loss_coef $CTC_loss_coef \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--lr ${lr} \
--n_epoch ${n_epoch} \
--lw_saliency ${lw_saliency} \
--lr_drop ${lr_drop} \
--attn ${attn} \
--s_window_size ${s_window_size} \
--f_window_size ${f_window_size} \
--f_dilation ${f_dilation} \
--f_model ${f_model} \
--sf_fuse ${sf_fuse} \
>> ${results_root}/${exp_id}_log.txt


attn=safa
s_window_size=3
f_window_size=13
f_dilation=2
f_model=64
sf_fuse=concat

exp_id=ws_${s_window_size}_${f_window_size}_d_${f_dilation}_fm${f_model}_fs_${sf_fuse}_seed_${seed}

CUDA_VISIBLE_DEVICES=${gpunum} PYTHONPATH=$PYTHONPATH:. python tr_detr/train.py \
--seed $seed \
--label_loss_coef $label_loss_coef \
--VTC_loss_coef $VTC_loss_coef \
--CTC_loss_coef $CTC_loss_coef \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--lr ${lr} \
--n_epoch ${n_epoch} \
--lw_saliency ${lw_saliency} \
--lr_drop ${lr_drop} \
--attn ${attn} \
--s_window_size ${s_window_size} \
--f_window_size ${f_window_size} \
--f_dilation ${f_dilation} \
--f_model ${f_model} \
--sf_fuse ${sf_fuse} \
>> ${results_root}/${exp_id}_log.txt

done

