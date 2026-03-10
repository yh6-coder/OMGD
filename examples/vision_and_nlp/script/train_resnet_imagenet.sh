model=resnet18
dataset=imagenet
epochs=100
start_epoch=0
bsz=128
num_workers=16
lr=0.1
stype=random_reshuffling
seed=888
mask_type=wor

base_dir=$(pwd)

run_cmd="torchrun --nproc_per_node=4 --standalone train_resnet_imagenet.py --model=${model} \
        --dataset=${dataset} \
        --data_path=/nvme1/share/imagenet/ILSVRC2012 \
        --start_epoch=${start_epoch} \
        --epochs=${epochs} \
        --batch_size=${bsz} \
        --lr=${lr} \
        --shuffle_type=${stype} \
        --seed=${seed} \
        --momentum=0.9 \
        --weight_decay=1e-4 \
        --use_tensorboard \
        --tensorboard_path=${base_dir} \
        --num_workers=${num_workers} \
        --transforms_json=${base_dir}/jsons/imagenet.json \
        --mask_type=${mask_type} \
        --r=0.5 \
        --checkpoint_freq=30 
        "

echo ${run_cmd}
eval ${run_cmd}
