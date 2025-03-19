import os

import jax
import jax.distributed
import jax.numpy as jnp


_id = int(os.uname()[1][-1]) - 1
id = int(os.environ.get("SLURM_PROCID", _id))

# rank = jax.process_index()
# size = jax.process_count()


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

    jax.distributed.initialize(
        coordinator_address=f"{master}:1234",
        num_processes=nprocs,
        process_id=id,
    )


def show():

    # Each node will create an array of ones with size == number of local GPUs
    xs = jax.numpy.ones(jax.local_device_count())
    # The psum is performed over all mapped GPU devices across the cluster
    y = jax.pmap(lambda x: jax.lax.psum(x, "i"), axis_name="i")(xs)

    print(jax.process_index())

    if id == 0:
        print(jax.devices())
        print("on MN")
        print(
            f"\nProcess: {jax.process_index()}\nGlobal device count: {jax.device_count()}\n"
            f"\nLocal device count: {jax.local_device_count()}\n"
            f"\nInput: {xs}\nOutput: {y}\n"
        )

