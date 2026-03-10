export CUDA_VISIBLE_DEVICES=0,1,2,3
model=vit-base-patch16-224-in21k
dataset=cifar10
epochs=100
bsz=128
lr=1e-4
stype=random_reshuffling
seed=555
mask_type=lisa_wor
r=0.5
sampling_layers=3
sampling_period=5

base_dir=$(pwd)

checkpoint_dir="${base_dir}/checkpoints/${model}_${dataset}_seed${seed}_mask${mask_type}_r${r}"
mkdir -p "${checkpoint_dir}"

resume_dir=""

run_cmd="torchrun --nproc_per_node=4 --standalone fine-tuning_vit.py \
        --model=${model} \
        --dataset=${dataset} \
        --data_path=/nvme1/share \
        --epochs=${epochs} \
        --batch_size=${bsz} \
        --lr=${lr} \
        --shuffle_type=${stype} \
        --seed=${seed} \
        --momentum=0.9 \
        --weight_decay=1e-4 \
        --use_tensorboard \
        --tensorboard_path=${base_dir} \
        --num_workers=8 \
        --transforms_json=${base_dir}/jsons/cifar10.json \
        --mask_type=${mask_type} \
        --r=${r} \
        --warm_up=0 \
        --checkpoint_path=${checkpoint_dir} \
        --checkpoint_freq=20 \
        --resume=${resume_dir} \
        --print_freq=2 \
        --sampling_period=${sampling_period} \
        --sampling_layers=${sampling_layers} \
        --test_batch_size=512
        "

echo ${run_cmd}
eval ${run_cmd}
