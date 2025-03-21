import os
import argparse
import json
import random
from functools import partial
import datetime 

import numpy as np

import torch
import torch.utils.data.distributed
import torch.nn as nn
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader, SequentialSampler


import unik3d.datasets as datasets
from unik3d.datasets import (DistributedSamplerNoDuplicate, collate_fn)
from unik3d.models import UniK3D
from unik3d.utils import (is_main_process, validate)
from unik3d.utils.distributed import (create_local_process_group,
                                      local_broadcast_process_authkey,
                                      setup_multi_processes, setup_slurm)


def main_worker(args: argparse.Namespace, config_file: str | None = None):
    with open(config_file, "r") as f:
        if is_main_process(): print("Config: ", config_file)
        config = json.load(f)

    if not args.distributed:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        args.rank = 0
        args.world_size = 1
    else:
        # initializes the distributed backend which will take care of synchronizing nodes/GPUs
        setup_multi_processes(config)
        is_slurm = "SLURM_PROCID" in os.environ
        if is_slurm:
            setup_slurm("nccl", port=args.master_port)
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = device = int(os.environ["LOCAL_RANK"])
        device = args.local_rank
        if not is_slurm:
            dist.init_process_group(backend="nccl", rank=args.rank, world_size=args.world_size, timeout=datetime.timedelta(seconds=30*60))
            torch.cuda.set_device(device)
        create_local_process_group()
        local_broadcast_process_authkey()

        print(f"Start running DDP on {args.rank}.")

    ##############################
    ########### MODEL ############
    ##############################
    # Build model
    version = config_file.split("/")[-1].split(".")[0]
    model = UniK3D.from_pretrained(f"lpiccinelli/unik3d-{version}")
    # you can load yours here
    # model.load_pretrained(config["training"]["pretrained"])
    model.eval()
    model.pixel_decoder.camera_gt = args.camera_gt

    print(f"MODEL: {model.__class__.__name__} at {model.device}")
    torch.cuda.empty_cache()
    model = model.to(device)
    
    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if any([p.requires_grad for p in model.parameters()]) and device != "cpu":
            model = DistributedDataParallel(model, find_unused_parameters=False, device_ids=[device], output_device=device)

    ##############################
    ########## DATASET ###########
    ##############################
    # Datasets loading
    resize_method = config["data"].get("resize_method", "hard") 
    crop = config["data"].get("crop", "garg")
    augmentations_db = config["data"].get("augmentations", {})
    image_shape = config["data"]["image_shape"]
    datasets_names = config["data"]["val_datasets"]

    if is_main_process():
        print("Loading validation datasets...")
    val_datasets = {}
    for dataset in datasets_names:
        val_dataset: datasets.BaseDataset = getattr(datasets, dataset)
        val_datasets[dataset] = val_dataset(
            image_shape=image_shape,
            split_file=val_dataset.test_split,
            test_mode=True,
            crop=crop,
            augmentations_db=augmentations_db,
            normalize=config["data"].get("normalization", "imagenet"),
            resize_method=resize_method,
            shape_constraints=config["data"].get("shape_constraints", {}),
            num_frames=1,
            mini=1.0,
        )

    # Dataset samplers, create distributed sampler pinned to rank
    if args.distributed:
        valid_samplers = {k: DistributedSamplerNoDuplicate(v, num_replicas=args.world_size, rank=args.rank, shuffle=False, drop_last=False) for k,v in val_datasets.items()}
    else:
        valid_samplers = {k: SequentialSampler(v) for k,v in val_datasets.items()}

    # Dataset loader
    val_batch_size = 1
    num_workers = 4
    val_loaders = {
        name_dataset: DataLoader(
            dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=valid_samplers[name_dataset],
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, is_batched=False)
        ) for name_dataset, dataset in val_datasets.items()
    }
    
    is_shell = int(os.environ.get("SHELL_JOB", "0"))
    if is_main_process(): 
        print("shell job?", is_shell)
    context = torch.autocast(device_type="cuda" if device != "cpu" else "cpu", enabled=True, dtype=torch.float16)

    model.eval()
    if is_main_process(): 
        print("Start validation...")
    with torch.no_grad():
        stats = validate(
            model,
            test_loaders=val_loaders,
            step=0,
            run_id="dummy",
            context=context,
            idxs=[0],
        )
    
    stats = {k: v for k,v in stats.items() if k in config["data"]["val_datasets"]}
    if is_main_process():
        with open(args.save_path, "w") as f:
            json.dump(stats, f, indent=4)



if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='Training script', conflict_handler='resolve')
    parser.add_argument("--master-port", type=str, default=29400)
    parser.add_argument("--distributed",  action="store_true")
    parser.add_argument("--local_rank",  type=int, default=0)
    parser.add_argument("--config-file", type=str, default="./configs/eval/vitl.json")
    parser.add_argument("--camera-gt",  action="store_true")
    parser.add_argument("--save-path", type=str, default="./unik3d.json")
    parser.add_argument("--dataroot", type=str, required=True)

    args = parser.parse_args()
    metrics_all = []
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    stats = main_worker(args, args.config_file)

