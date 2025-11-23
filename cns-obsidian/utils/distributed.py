import os
import time
import torch
import socket

import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    StateDictType,
)


def setup_distributed():
    print(f"{time.ctime()}: Starting distributed setup")

    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])

    print(f"{time.ctime()}: Rank {rank}: Determining master address")
    if "SLURM_LAUNCH_NODE_IPADDR" in os.environ:
        master_addr = os.environ["SLURM_LAUNCH_NODE_IPADDR"]
    else:
        master_addr = socket.gethostbyname(
            os.environ["SLURM_NODELIST"].split(",")[0]
        )

    print(f"{time.ctime()}: Rank {rank}: Master address is {master_addr}")

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = "29500"  # Use a fixed port

    print(
        f"{time.ctime()}: Rank {rank}: Environment variables set. Initializing process group"
    )

    # Initialize the process group
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:29500",
        rank=rank,
        world_size=world_size,
    )

    print(f"{time.ctime()}: Rank {rank}: Process group initialized")

    torch.cuda.set_device(rank % torch.cuda.device_count())

    print(
        f"{time.ctime()}: Rank {rank}: CUDA device set to {torch.cuda.current_device()}"
    )
    print(
        f"{time.ctime()}: Rank {rank}: Setup complete. World size: {dist.get_world_size()}"
    )


def kill_distributed():
    print(f"{time.ctime()}: Killing distributed setup")
    dist.destroy_process_group()


def save_model_distributed(model, optimizer, scheduler, epoch, loss, checkpoints_dir):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        model_state = model.state_dict()
    if dist.get_rank() == 0:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": loss,
            },
            f"{checkpoints_dir}/checkpoint_{epoch}.pt",
        )
