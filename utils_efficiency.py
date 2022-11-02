import torch
import torch.nn as nn
import numpy as np
from termcolor import cprint


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_efficiency_metrics(interval_train_epochs,
                               num_batches_train_epochs,
                               interval_valid_epochs,
                               num_batches_valid_epochs,
                               total_epoch_count,
                               model: nn.Module,
                               device=None):

    dt_init_start = None  # not support for this function

    cprint("------------------", "yellow")
    total_train_time = sum(interval_train_epochs)
    dt_train_epoch = np.mean(interval_train_epochs)
    dt_train_batch = sum(interval_train_epochs) / sum(num_batches_train_epochs)
    total_valid_time = sum(interval_valid_epochs)
    dt_valid_epoch = np.mean(interval_valid_epochs)
    dt_valid_batch = sum(interval_valid_epochs) / sum(num_batches_valid_epochs)

    cprint(f"- total_epoch: {total_epoch_count}", "yellow")

    cprint(f"- total_train_time: {total_train_time}", "yellow")
    cprint(f"- time / train_epoch: {dt_train_epoch}", "yellow")
    cprint(f"- time / train_batch: {dt_train_batch}", "yellow")

    cprint(f"- total_valid_time: {total_valid_time}", "yellow")
    cprint(f"- time / valid_epoch: {dt_valid_epoch}", "yellow")
    cprint(f"- time / valid_batch: {dt_valid_batch}", "yellow")

    # Memory and parameters
    num_parameters = count_parameters(model)
    max_memory_reserved = torch.cuda.max_memory_reserved(device)
    max_memory_allocated = torch.cuda.max_memory_allocated(device)
    cprint(f"\nSummary as Table --- {model.__class__.__name__}", "yellow")
    print("\t".join(str(t) for t in [
        num_parameters, max_memory_reserved, max_memory_allocated,
        dt_init_start, total_epoch_count,
        total_train_time, dt_train_epoch, dt_train_batch,
        total_valid_time, dt_valid_epoch, dt_valid_batch,
    ]))
