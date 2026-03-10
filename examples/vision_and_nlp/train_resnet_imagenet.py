import os
import random
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import torchvision
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter

from arguments import get_args
from utils import train, validate, Timer, build_task_name, ImageFolderLMDB, CycleBatchSampler
from constants import _RANDOM_RESHUFFLING_, _SHUFFLE_ONCE_, _STALE_GRAD_SORT_, _ZEROTH_ORDER_SORT_, _FRESH_GRAD_SORT_, \
    _IMAGENET_

from tqdm import trange

logger = logging.getLogger(__name__)

def main():
    args = get_args()

    if args.seed == 0:
        args.seed = random.randint(0, 10000)
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger.info(f"Using random seed {args.seed} for random and torch module.")

    rank = int(os.environ.get('RANK', -1))
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if rank != -1: 
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.use_cuda = torch.cuda.is_available()
    logger.info(f"Using GPU: {args.use_cuda}")

    timer = Timer(verbosity_level=1, use_cuda=args.use_cuda)

    criterion = torch.nn.CrossEntropyLoss()
    if args.use_cuda:
        criterion.cuda()
    logger.info(f"Using Cross Entropy Loss for classification.")
    
    # model = torch.nn.DataParallel(torchvision.models.__dict__[args.model](pretrained=args.pretrained))
    if args.pretrained and args.start_epoch == 90:
        weights = torchvision.models.ResNet18_Weights.DEFAULT  # 或 IMAGENET1K_V1
    else:
        weights = None
    model = torchvision.models.__dict__[args.model](weights=weights)
    # if args.use_cuda:
    #     model.cuda()
    model = model.to(device)
    model_dimen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # model = VisionModel(args, model, criterion)
    logger.info(f"Using model: {args.model} with dimension: {model_dimen}.")

    if rank != -1:
        ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        optimizer = torch.optim.SGD(params=ddp_model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
        
    logger.info(f"Using optimizer SGD with hyperparameters: learning rate={args.lr}; momentum={args.momentum}; weight decay={args.weight_decay}.")

    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[30, 60, 90], last_epoch=args.start_epoch-1)
    
    logger.info(f"Using dataset: {args.dataset}")
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            args.start_epoch = checkpoint['epoch'] + 1 
            if 'lr_scheduler_state_dict' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            print(f"Resumed from epoch {checkpoint['epoch']}, val_acc: {checkpoint.get('val_acc', 'N/A')}")
        else:
            print(f"Checkpoint {args.resume} not found, starting from scratch")
            

    loaders = {}
    shuffle_flag = True if args.shuffle_type in [_RANDOM_RESHUFFLING_, _ZEROTH_ORDER_SORT_, _FRESH_GRAD_SORT_] else False
    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'val')
    
    start_time = time.time()
    
    train_transforms = transforms.Compose([
        transforms.Resize(256),                
        transforms.RandomResizedCrop(224),     
        transforms.RandomHorizontalFlip(),     
        transforms.ToTensor(),                 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                            std=[0.229, 0.224, 0.225])
    ])


    test_transforms = transforms.Compose([
        transforms.Resize(256),               
        transforms.CenterCrop(224),         
        transforms.ToTensor(),                
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])


    trainset = datasets.ImageFolder(root = traindir, transform = train_transforms)
    testset = datasets.ImageFolder(root = valdir, transform = test_transforms)
    end_time = time.time()
    print(f'Finish creating trainset and testset, time cost: {end_time - start_time} seconds')

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


    # for epoch in range(args.start_epoch, args.epochs):
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
        
        # Checkpoint 保存逻辑
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
        dist.destroy_process_group() 

    logger.info(f"Finish training!")

if __name__ == '__main__':
    main()
