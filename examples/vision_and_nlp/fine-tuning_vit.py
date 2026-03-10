import os
import random
import torch
import logging
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as T
import time
from tensorboardX import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import ViTForImageClassification

from models import resnet
from arguments import get_args
from utils import train, validate, Timer, build_task_name, CycleBatchSampler
from constants import _RANDOM_RESHUFFLING_, _SHUFFLE_ONCE_, _STALE_GRAD_SORT_, _ZEROTH_ORDER_SORT_, _FRESH_GRAD_SORT_, \
    _CIFAR10_, _CIFAR100_
from tqdm import trange

from sift import SIFT

logger = logging.getLogger(__name__)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29520'  
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    dist.destroy_process_group()

model_configs = {
    'resnet20': [3, 3, 3],
    'resnet32': [5, 5, 5],
    'resnet44': [7, 7, 7],
    'resnet56': [9, 9, 9],
    'resnet110': [18, 18, 18],
    'resnet1202': [200, 200, 200]
}

def main():
    args = get_args()

    if args.seed == 0:
        args.seed = random.randint(0, 10000)
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # dist.init_process_group(backend='nccl')
    rank = int(os.environ.get('RANK', -1))
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    print(rank, local_rank, world_size)
    
    if rank != -1: 
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using random seed {args.seed} for random and torch module.")

    args.use_cuda = torch.cuda.is_available()

    if args.dataset == _CIFAR100_:
        num_head = 100
    elif args.dataset == _CIFAR10_:
        num_head = 10
    elif args.dataset == 'imagenet':
        num_head = 1000
    else:
        raise ValueError("Unsupported dataset")
    
    model_path = "./models/google/vit-base-patch16-224-in21k" 
    model = ViTForImageClassification.from_pretrained(
            model_path,
            num_labels=num_head, 
            ignore_mismatched_sizes=True
            )   
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    if args.use_cuda:
        criterion.cuda()
    logger.info(f"Using Cross Entropy Loss for classification.")

    main_worker(rank, local_rank, world_size, args, model, criterion, device)

def main_worker(rank, local_rank, world_size, args, model, criterion, device):

    if rank != -1:
        find_unused_parameters = False if args.mask_type in ['sift'] else True 
        ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=find_unused_parameters)
        if args.mask_type == 'sift':
            sift = SIFT(ddp_model, sparse_rate=0.2, sparse_module=[
                            "attention.attention.query",
                            "attention.attention.key",
                            "attention.attention.value",
                            "attention.output.dense",
                            "intermediate.dense",
                            "output.dense"],
                        exception=['classifier', 'embeddings'],
                        grad_acc=1)
            sift.print_trainable_parameters()
            optimizer_grouped_parameters = [
            {
                "params": [p for n, p in sift.named_parameters_in_optimizer()] ,
                "weight_decay": args.weight_decay,
            }   
            ]
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                lr=args.lr)
        elif args.mask_type == 'golore':
            from Golore.peft_pretraining.GoLore import ReLoRaModel
            from Golore.peft_pretraining.adamw import AdamW as GoloreAdamW
            ddp_model = ReLoRaModel(ddp_model,
                                r=128,  
                                lora_dropout=0.0,
                                target_modules=["attention", "mlp"], 
                                keep_original_weights=True,
                                scale=1.0,)
            ddp_model = ddp_model.to(device)  
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in ddp_model.named_parameters() if  p.requires_grad],
                    "weight_decay": args.weight_decay,
                }]
            optimizer = GoloreAdamW(optimizer_grouped_parameters,
                                lr=args.lr,
                                weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(params=ddp_model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay)
    else:
        ddp_model = model
        if args.mask_type == 'sift':
            sift = SIFT(model, sparse_rate=0.5, sparse_module=[
                                "attention.attention.query",
                                "attention.attention.key",
                                "attention.attention.value",
                                "attention.output.dense",
                                "intermediate.dense",
                                "output.dense",
                            ])
            sift.print_trainable_parameters()
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in sift.named_parameters_in_optimizer()] ,
                    "weight_decay": args.weight_decay,
                }   
                ]
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                    lr=args.lr)
        else:
            optimizer = torch.optim.AdamW(params=ddp_model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay)

    timer = Timer(verbosity_level=1, use_cuda=args.use_cuda)

    model_dimen = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Using model: {args.model} with dimension: {model_dimen}.")
    logger.info(f"Using optimizer Adam with hyperparameters: learning rate={args.lr}.")
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)
    logger.info(f"Using dataset: {args.dataset}")
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            args.start_epoch = checkpoint['epoch'] + 1  # 从下一个 epoch 开始
            if 'lr_scheduler_state_dict' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            print(f"Resumed from epoch {checkpoint['epoch']}")
        else:
            print(f"Checkpoint {args.resume} not found, starting from scratch")

    loaders = {}
    shuffle_flag = True if args.shuffle_type in [_RANDOM_RESHUFFLING_, _ZEROTH_ORDER_SORT_, _FRESH_GRAD_SORT_] else False
    data_path = args.data_path
    cifar_mean = (0.485, 0.456, 0.406)
    cifar_std  = (0.229, 0.224, 0.225)

    train_transform = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.autoaugment.RandAugment(),
        T.ToTensor(),
        T.Normalize(cifar_mean, cifar_std),
        ])

    test_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(cifar_mean, cifar_std),
    ])

    if args.dataset == _CIFAR10_:
        trainset = datasets.CIFAR10(root=data_path, train=True, transform=train_transform)
        testset  = datasets.CIFAR10(root=data_path, train=False, transform=test_transform)
    elif args.dataset == _CIFAR100_:
        trainset = datasets.CIFAR100(root=data_path, train=True, transform=train_transform)
        testset  = datasets.CIFAR100(root=data_path, train=False, transform=test_transform)
    elif args.dataset == 'imagenet':
        trainset = datasets.ImageFolder(root = os.path.join(args.data_path, 'train'), transform = train_transform)
        testset = datasets.ImageFolder(root = os.path.join(args.data_path, 'val'), transform = test_transform)
    else:
        raise NotImplementedError("This script is for CIFAR datasets. Please input cifar10 or cifar100 in --dataset.")
    
    if rank != -1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=shuffle_flag)
        loaders['train'] = torch.utils.data.DataLoader(trainset,
                                                        batch_size=args.batch_size,
                                                        sampler=train_sampler,
                                                        persistent_workers=False,
                                                        num_workers=args.num_workers,
                                                        pin_memory=False)
    else:
        loaders['train'] = torch.utils.data.DataLoader(trainset,
                                                        batch_size=args.batch_size,
                                                        shuffle=shuffle_flag,
                                                        persistent_workers=False,
                                                        num_workers=args.num_workers,
                                                        pin_memory=False)
        
    loaders['val'] = torch.utils.data.DataLoader(testset,
                                                    batch_size=args.test_batch_size,
                                                    shuffle=False,
                                                    num_workers=args.num_workers, 
                                                    pin_memory=False)

    # Epoch-wise data ordering
    if args.shuffle_type in [_RANDOM_RESHUFFLING_, _SHUFFLE_ONCE_]:
        sorter = None

    args.task_name = build_task_name(args)
    logger.info(f"Creating task name as: {args.task_name}.")

    if args.use_tensorboard:
        tb_path = os.path.join(args.tensorboard_path, 'runs', args.task_name)
        logger.info(f"Streaming tensorboard logs to path: {tb_path}.")
        tb_logger = SummaryWriter(tb_path)
    else:
        tb_logger = None
        logger.info(f"Disable tensorboard logs currently.")
    
    for epoch in trange(args.start_epoch, args.epochs, desc="Epoch Progress", disable=(rank not in [-1, 0])):

        train(args=args,
            loader=loaders['train'],
            model=ddp_model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            tb_logger=tb_logger if rank in [-1, 0] else None,
            timer=timer,
            sorter=sorter,
            rank=rank,
            world_size=world_size)
        
        lr_scheduler.step()

        # evaluate on validation set
        if rank in [-1, 0]:
            validate(args=args,
                loader=loaders['val'],
                model=model,
                criterion=criterion,
                epoch=epoch,
                tb_logger=tb_logger)
        
        # Checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
            if rank in [-1, 0]:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'args': args
                }
                checkpoint_file = os.path.join(args.checkpoint_path, f'checkpoint_epoch_{epoch}.pth')
                torch.save(checkpoint, checkpoint_file)
                print(f"Checkpoint saved at epoch {epoch}: {checkpoint_file}")

    tb_logger.close()

    if rank != -1:
        cleanup_distributed()

    logger.info(f"Finish training!")

if __name__ == '__main__':
    main()
