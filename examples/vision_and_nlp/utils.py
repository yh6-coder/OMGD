import os
import torch
import time
import pickle
import logging
import lmdb
from contextlib import contextmanager
from io import StringIO
from constants import _STALE_GRAD_SORT_, _ZEROTH_ORDER_SORT_, _FRESH_GRAD_SORT_, _MNIST_
import torch.utils.data as data
from visionmodel import accuracy

from tqdm import trange

import torch
import torch.distributed as dist

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(args,
        loader,
        model,
        criterion,
        optimizer,
        epoch,
        tb_logger,
        timer=None,
        sorter=None,
        rank=-1,
        world_size=1):
    
    if rank in [-1, 0]:
        losses = AverageMeter()
        top1 = AverageMeter()

    model.train()
    params = list(model.parameters())
    P = len(params)

    # Simplified mask logic
    if args.mask_type == 'iid':
        # Each epoch, randomly select r * P parameters
        r = args.r
        k = int(round(r * P))
        idxs = torch.randperm(P)[:k]
        mask = torch.zeros(P, dtype=torch.bool)
        mask[idxs] = True
    elif args.mask_type == 'wor':
        # Pre-generate non-overlapping masks if not done
        if not hasattr(args, 'wor_masks') or epoch == 0:
            r = args.r
            num_masks = int(round(1 / r))
            args.wor_masks = []
            indices = torch.randperm(P)
            k = P // num_masks
            for j in range(num_masks):
                start = j * k
                end = (j + 1) * k if j < num_masks - 1 else P
                mask = torch.zeros(P, dtype=torch.bool)
                mask[indices[start:end]] = True
                args.wor_masks.append(mask)
        # Select mask for this epoch
        mask_idx = epoch % len(args.wor_masks)
        mask = args.wor_masks[mask_idx]
    elif args.mask_type == 'lisa':
        if not hasattr(args, 'layer_wise_masks') or (epoch % args.sampling_period == 0):
            named_params = list(model.named_parameters())
            layer_groups = {}
            for idx, (name, _) in enumerate(named_params):
                if name.startswith('module.vit.encoder.layer'):
                    layer_num = int(name.split('.')[4])  # Extract layer number, e.g., '0', '11'
                    if layer_num not in layer_groups:
                        layer_groups[layer_num] = []
                    layer_groups[layer_num].append(idx)
            middle_layers = list(layer_groups.keys())
            selected_indices = torch.randperm(len(middle_layers))[:args.sampling_layers]
            mask = torch.zeros(P, dtype=torch.bool)
            for sel_idx in selected_indices:
                layer_num = middle_layers[sel_idx]
                for idx in layer_groups[layer_num]:
                    mask[idx] = True
            # Head/tail always True
            for idx, (name, _) in enumerate(named_params):
                if not name.startswith('module.vit.encoder.layer'):
                    mask[idx] = True
            args.layer_wise_masks = [mask]
            if rank in [-1, 0]:
                print(middle_layers)
                print(selected_indices)
        mask = args.layer_wise_masks[0]
    elif args.mask_type == 'lisa_wor':
        if not hasattr(args, 'used_layers'):
            args.used_layers = set()
        if not hasattr(args, 'layer_wise_masks') or (epoch % args.sampling_period == 0):
            named_params = list(model.named_parameters())
            layer_groups = {}
            for idx, (name, _) in enumerate(named_params):
                if name.startswith('module.vit.encoder.layer'):
                    layer_num = int(name.split('.')[4])
                    if layer_num not in layer_groups:
                        layer_groups[layer_num] = []
                    layer_groups[layer_num].append(idx)
            middle_layers = list(layer_groups.keys())
            if len(args.used_layers) == len(middle_layers):
                args.used_layers = set()
            available = [l for l in middle_layers if l not in args.used_layers]
            selected_indices = torch.randperm(len(available))[:args.sampling_layers]
            selected_layers = [available[i] for i in selected_indices]
            args.used_layers.update(selected_layers)
            mask = torch.zeros(P, dtype=torch.bool)
            for layer_num in selected_layers:
                for idx in layer_groups[layer_num]:
                    mask[idx] = True
            # Head/tail always True
            for idx, (name, _) in enumerate(named_params):
                if not name.startswith('module.vit.encoder.layer'):
                    mask[idx] = True
            args.layer_wise_masks = [mask]
            if rank in [-1, 0]:
                print(selected_layers)
                print(args.used_layers)
        mask = args.layer_wise_masks[0]
        
    else:
        # No mask, all parameters trainable
        mask = torch.ones(P, dtype=torch.bool)

    # Set requires_grad based on mask
    if epoch >= args.warm_up:
        for j, param in enumerate(params):
            param.requires_grad = mask[j].item()

    # Training loop (show progress with trange)
    loader_iter = iter(loader)
    num_batches = len(loader)
    for i in trange(num_batches, desc=f"Train Epoch {epoch}", leave=False, disable=(rank not in [-1, 0])):
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        
        with timer("forward pass", epoch=epoch):
            # loss, prec1, cur_batch_size = model(batch)
            input_var, target_var = batch
            if args.use_cuda:
                input_var = input_var.cuda()
                target_var = target_var.cuda()
            
            output = model(input_var) 
            if hasattr(output, 'logits'):
                output = output.logits 
            loss = criterion(output, target_var)  
            prec1 = accuracy(output.data, target_var)[0]
            cur_batch_size = input_var.size(0)
        
        with timer("backward pass", epoch=epoch):
            optimizer.zero_grad()
            loss.backward()
        
        # Apply gradient scaling for masked parameters
        if epoch >= args.warm_up:
            if args.mask_type in ['iid', 'wor']:
                for j, param in enumerate(params):
                    if mask[j].item() and param.grad is not None:
                        param.grad.div_(args.r)
            elif args.mask_type in ['lisa_wor']:
                named_params = list(model.named_parameters())
                for j, param in enumerate(params):
                    name = named_params[j][0]
                    # Only scale gradients for encoder layers that are masked and active
                    if mask[j].item() and param.grad is not None and name.startswith('module.vit.encoder.layer'):
                        # if rank in [-1, 0]: print(name)
                        param.grad.mul_(12/args.sampling_layers)
        optimizer.step()

        loss = loss.float()
        if rank != -1:
                # All-reduce loss and prec1
                dist.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
                dist.all_reduce(prec1, op=torch.distributed.ReduceOp.SUM)
                loss /= world_size
                prec1 /= world_size
                dist.barrier()
        if rank in [-1, 0]:
            losses.update(loss.item(), cur_batch_size)
            top1.update(prec1.item(), cur_batch_size)

        # if i % args.print_freq == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        #               epoch, i, len(loader), loss=losses, top1=top1))

        if args.use_tensorboard and rank in [-1, 0]:
            global_step = epoch * len(loader) + i
            tb_logger.add_scalar('train/step_loss', loss.item(), global_step)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    avg_norm = param.norm().item() / param.numel()
                    tb_logger.add_scalar(f'param_norm/{name}', avg_norm, global_step)

    if args.use_tensorboard and rank in [-1, 0]:
        tb_logger.add_scalar('train/accuracy', top1.avg, epoch)
        tb_logger.add_scalar('train/loss', losses.avg, epoch)
        total_time = timer.totals["forward pass"] + timer.totals["backward pass"]
        tb_logger.add_scalar('train_time/accuracy', top1.avg, total_time)
        tb_logger.add_scalar('train_time/loss', losses.avg, total_time)
        tb_logger.add_scalar('forward_time/total_time', timer.totals["forward pass"], epoch)
        tb_logger.add_scalar('backward_time/total_time', timer.totals["backward pass"], epoch)
    
    return

def validate(args, loader, model, criterion, epoch, tb_logger):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            # loss, prec1, cur_batch_size = model(batch)
            input_var, target_var = batch
            if args.use_cuda:
                input_var = input_var.cuda()
                target_var = target_var.cuda()
            
            output = model(input_var)  
            if hasattr(output, 'logits'):
                output = output.logits  
            loss = criterion(output, target_var)  
            prec1 = accuracy(output.data, target_var)[0]
            cur_batch_size = input_var.size(0)
            loss = loss.float()

            # measure accuracy and record loss
            losses.update(loss.item(), cur_batch_size)
            top1.update(prec1.item(), cur_batch_size)

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(loader), loss=losses,
                          top1=top1))
    if args.use_tensorboard:
        tb_logger.add_scalar('test/accuracy', top1.avg, epoch)
        tb_logger.add_scalar('test/loss', losses.avg, epoch)

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return



class Timer:
    """
    Timer for PyTorch code
    Comes in the form of a contextmanager:
    Example:
    >>> timer = Timer()
    ... for i in range(10):
    ...     with timer("expensive operation"):
    ...         x = torch.randn(100)
    ... print(timer.summary())
    """

    def __init__(self, verbosity_level=1, skip_first=True, use_cuda=True):
        self.verbosity_level = verbosity_level
        #self.log_fn = log_fn if log_fn is not None else self._default_log_fn
        self.skip_first = skip_first
        self.cuda_available = torch.cuda.is_available() and use_cuda

        self.reset()

    def reset(self):
        """Reset the timer"""
        self.totals = {}  # Total time per label
        self.first_time = {}  # First occurrence of a label (start time)
        self.last_time = {}  # Last occurence of a label (end time)
        self.call_counts = {}  # Number of times a label occurred

    @contextmanager
    def __call__(self, label, epoch=-1.0, verbosity=1):
        # Don't measure this if the verbosity level is too high
        if verbosity > self.verbosity_level:
            yield
            return

        # Measure the time
        self._cuda_sync()
        start = time.time()
        yield
        self._cuda_sync()
        end = time.time()

        # Update first and last occurrence of this label
        if label not in self.first_time:
            self.first_time[label] = start
        self.last_time[label] = end

        # Update the totals and call counts
        if label not in self.totals and self.skip_first:
            self.totals[label] = 0.0
            del self.first_time[label]
            self.call_counts[label] = 0
        elif label not in self.totals and not self.skip_first:
            self.totals[label] = end - start
            self.call_counts[label] = 1
        else:
            self.totals[label] += end - start
            self.call_counts[label] += 1

        #if self.call_counts[label] > 0:
        #    # We will reduce the probability of logging a timing
        #    # linearly with the number of time we have seen it.
        #    # It will always be recorded in the totals, though.
        #    if np.random.rand() < 1 / self.call_counts[label]:
        #        self.log_fn(
        #            "timer", {"epoch": epoch, "value": end - start}, {"event": label}
        #        )

    def summary(self):
        """
        Return a summary in string-form of all the timings recorded so far
        """
        if len(self.totals) > 0:
            with StringIO() as buffer:
                total_avg_time = 0
                print("--- Timer summary ------------------------", file=buffer)
                print("  Event   |  Count | Average time |  Frac.", file=buffer)
                for event_label in sorted(self.totals):
                    total = self.totals[event_label]
                    count = self.call_counts[event_label]
                    if count == 0:
                        continue
                    avg_duration = total / count
                    total_runtime = (
                        self.last_time[event_label] - self.first_time[event_label]
                    )
                    runtime_percentage = 100 * total / total_runtime
                    total_avg_time += avg_duration if "." not in event_label else 0
                    print(
                        f"- {event_label:30s} | {count:6d} | {avg_duration:11.5f}s | {runtime_percentage:5.1f}%",
                        file=buffer,
                    )
                print("-------------------------------------------", file=buffer)
                event_label = "total_averaged_time"
                print(
                    f"- {event_label:30s}| {count:6d} | {total_avg_time:11.5f}s |",
                    file=buffer,
                )
                print("-------------------------------------------", file=buffer)
                return buffer.getvalue()

    def _cuda_sync(self):
        """Finish all asynchronous GPU computations to get correct timings"""
        if self.cuda_available:
            torch.cuda.synchronize()

    def _default_log_fn(self, _, values, tags):
        label = tags["label"]
        epoch = values["epoch"]
        duration = values["value"]
        print(f"Timer: {label:30s} @ {epoch:4.1f} - {duration:8.5f}s")


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_data(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pickle.dumps(obj)

## Helper functions for ImageNet
def folder2lmdb(spath, dpath, name="train", write_frequency=5000):
    directory = os.path.expanduser(os.path.join(spath, name))
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    dataset = ImageFolder(directory, loader=raw_reader)
    data_loader = DataLoader(dataset, num_workers=16, collate_fn=lambda x: x)

    lmdb_path = os.path.join(dpath, "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx, data in enumerate(data_loader):
        image, label = data[0]

        txn.put(u'{}'.format(idx).encode('ascii'), dumps_data((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_data(keys))
        txn.put(b'__len__', dumps_data(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()

class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_data(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        # im2arr = np.array(img)
        # im2arr = torch.from_numpy(im2arr)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
        # return im2arr, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'