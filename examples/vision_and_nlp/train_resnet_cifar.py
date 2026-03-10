import os
import random
import torch
import logging
import torchvision.datasets as datasets
import torchvision.transforms as T
from tensorboardX import SummaryWriter

from models import resnet
from arguments import get_args
from utils import train, validate, Timer, build_task_name, CycleBatchSampler
from constants import _RANDOM_RESHUFFLING_, _SHUFFLE_ONCE_, _STALE_GRAD_SORT_, _ZEROTH_ORDER_SORT_, _FRESH_GRAD_SORT_, \
    _CIFAR10_, _CIFAR100_

from tqdm import trange

logger = logging.getLogger(__name__)

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
    logger.info(f"Using random seed {args.seed} for random and torch module.")

    args.use_cuda = torch.cuda.is_available()
    logger.info(f"Using GPU: {args.use_cuda}")

    timer = Timer(verbosity_level=1, use_cuda=args.use_cuda)

    args.gpu_id = '0'
    device = torch.device(f'cuda:{args.gpu_id}' if args.use_cuda else 'cpu')
    logger.info(f"Using device: {device}.")

    criterion = torch.nn.CrossEntropyLoss()
    if args.use_cuda:
        #criterion.cuda()
        criterion.to(device)
    logger.info(f"Using Cross Entropy Loss for classification.")

    if args.dataset == _CIFAR100_:
        num_head = 100
    elif args.dataset == _CIFAR10_:
        num_head = 10
    else:
        raise ValueError("Unsupported dataset")
    if args.model in model_configs:
        num_blocks = model_configs[args.model]
        model = torch.nn.DataParallel(resnet.ResNet(resnet.BasicBlock, num_blocks, num_classes=num_head))
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    # model = torch.nn.DataParallel(resnet.__dict__[args.model])
    
    if args.use_cuda:
        #model.cuda()
        model.to(device)
    model_dimen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # model = VisionModel(args, model, criterion)
    logger.info(f"Using model: {args.model} with dimension: {model_dimen}.")

    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    logger.info(f"Using optimizer SGD with hyperparameters: learning rate={args.lr}; momentum={args.momentum}; weight decay={args.weight_decay}.")

    if args.dataset == _CIFAR10_:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[100, 150],
                                                            last_epoch=args.start_epoch-1)
    elif args.dataset == _CIFAR100_:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[60, 120, 150],
                                                            gamma=0.2,
                                                            last_epoch=args.start_epoch-1)
    else:
        raise NotImplementedError("This script is for CIFAR datasets. Please input cifar10 or cifar100 in --dataset.")
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
    data_path = os.path.join(args.data_path, "data")
    
    if args.dataset == _CIFAR10_:
        cifar_mean = (0.485, 0.456, 0.406)
        cifar_std  = (0.229, 0.224, 0.225)
        train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(cifar_mean, cifar_std)
        ])
    elif args.dataset == _CIFAR100_:
        cifar_mean = (0.507, 0.487, 0.441)
        cifar_std = (0.268, 0.257, 0.276)
        train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(cifar_mean, cifar_std),
        T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
        ])

    test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(cifar_mean, cifar_std),
    ])

    if args.dataset == _CIFAR10_:
        trainset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transform)
        testset  = datasets.CIFAR10(root=data_path, train=False, download=True, transform=test_transform)
    elif args.dataset == _CIFAR100_:
        trainset = datasets.CIFAR100(root=data_path, train=True, download=True, transform=train_transform)
        testset  = datasets.CIFAR100(root=data_path, train=False, download=True, transform=test_transform)
    else:
        raise NotImplementedError("This script is for CIFAR datasets. Please input cifar10 or cifar100 in --dataset.")
    
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
    print(args.task_name)

    #for epoch in range(args.start_epoch, args.epochs):
    for epoch in trange(args.start_epoch, args.epochs, desc="Epoch Progress"):
        train(args=args,
            loader=loaders['train'],
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            tb_logger=tb_logger,
            timer=timer,
            sorter=sorter)
        
        lr_scheduler.step()

        # evaluate on validation set
        validate(args=args,
                loader=loaders['val'],
                model=model,
                criterion=criterion,
                epoch=epoch,
                tb_logger=tb_logger)
        
        # Checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
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

    logger.info(f"Finish training!")

if __name__ == '__main__':
    main()
