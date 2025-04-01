import os
import sys

import builtins
import jax

import time

import jax
import jax.distributed
import jax.numpy as jnp


# _id = int(os.uname()[1][-1]) - 1
# id = int(os.environ.get("SLURM_PROCID", _id))

# rank = jax.process_index()
# size = jax.process_count()

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world = int(comm.Get_size())
rank = int(comm.Get_rank())

lrank = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
lworld = os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE")

def init():

    if "SLURM_PROCID" in os.environ:
        if id == 0:
            slurm_vars = {
                key: value
                for key, value in os.environ.items()
                if key.startswith("SLURM_")
            }
            for key, value in slurm_vars.items():
                print(f"{key}: {value}")

    if "SLURM_NODELIST" in os.environ:
        master = os.environ.get("SLURM_NODELIST")
        master = master if "[" not in master else master[:3] + master[4]
        nprocs = int(os.environ.get("SLURM_NNODES"))
    else:
        nodelist = ["gpu1", "gpu2"]
        master = nodelist[0]
        nprocs = len(nodelist)

    nodefile = os.environ.get("PBS_NODEFILE")
    with open(nodefile, "r") as file:
        nodes = [x.strip() for x in file.readlines() if x]

    id = int([x.split(".")[0] for x in nodes].index(os.uname()[1]))


    wait = (len(nodes) - id) / len(nodes)
    time.sleep(wait)

    jax.distributed.initialize(
        coordinator_address=f"{nodes[0]}:29500",
        num_processes=world,  # len(nodes),
        process_id=rank,  # id,
    )

    # jax.distributed.initialize(
    # coordinator_address=f"{master}:29500",
    # num_processes=nprocs,
    # process_id=id,
    # )

    if jax.process_index() != 0:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        # Optionally suppress Python warnings
        import warnings
        warnings.filterwarnings("ignore")

    print = dist_print

    print(f"Hello from process {rank}")
    print(rank, world, lrank, lworld)
    print(nodes)
    print({"id": id, "master": nodes[0], "nodes": len(nodes), "world": world})



def dist_print(*args, **kwargs):
    if jax.process_index() == 0:
        builtins.print(*args, **kwargs)


def show():

    print(jax.process_index())

    if id == 0:
        print(jax.devices())
        print("on MN")
        print(
            f"\nProcess: {jax.process_index()}\nGlobal device count: {jax.device_count()}\n"
            f"\nLocal device count: {jax.local_device_count()}\n"
            f"\nInput: {xs}\nOutput: {y}\n"
        )
