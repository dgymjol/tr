dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=test
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
lr_drop=400
lr=0.0001
lw_saliency=1.0
seed=2017
VTC_loss_coef=0.3
CTC_loss_coef=0.5
# use_txt_pos=True
label_loss_coef=4



gpunum=0

org_duration=150
clip_len=2

n_epoch=2
bsz=8


results_root=results_mem_tome

if [ ! -d $results_root ]; then
  mkdir -p $results_root
fi


tome_r=15

list="1 25  50"

for multi_num in $list
do

tome_r_=$(expr $tome_r \* $multi_num)
duration=$(expr $org_duration \* $multi_num)
max_v_l=$(expr $duration / $clip_len)

echo "multi_num : $multi_num"
echo "duration : $duration"
echo "max_v_l : $max_v_l"
echo "tome_r_ : $tome_r_"

train_path=data/qv_${multi_num}.jsonl


attn=va
window_size=9
dilation=3

exp_id=unmerging_${attn}_tome_r_${tome_r_}_mult_${multi_num}

CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=${gpunum} PYTHONPATH=$PYTHONPATH:. python tr_detr/mem_test.py \
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
--max_v_l ${max_v_l} \
--attn ${attn} \
--window_size ${window_size} \
--tome \
--tome_r ${tome_r_} \
--unmerge \
>> ${results_root}/${exp_id}_log.txt

done
