model=resnet20
dataset=cifar10
epochs=200
bsz=128
lr=0.1
stype=random_reshuffling
seed=222
num_workers=16
mask_type=wor

base_dir=$(pwd)

run_cmd="python train_resnet_cifar.py --model=${model} \
        --dataset=${dataset} \
        --data_path=./datasets \
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
        --transforms_json=${base_dir}/jsons/cifar10.json \
        --mask_type=${mask_type} \
        --r=0.5
        "

echo ${run_cmd}
eval ${run_cmd}
