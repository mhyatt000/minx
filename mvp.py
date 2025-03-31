import os
import time

import jax
import jax.distributed
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import nnx
from flax.nnx.transforms.deprecated import N
from jax.debug import visualize_array_sharding as vas
from jax.experimental import multihost_utils as mx
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec


from mpi4py import MPI

for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(var, None)

os.environ["NCCL_SOCKET_IFNAME"] = "hsn"


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world = int(comm.Get_size())
rank = int(comm.Get_rank())

nodefile = os.environ.get("PBS_NODEFILE")
with open(nodefile, "r") as file:
    nodes = [x.strip() for x in file.readlines() if x]

ngpus: int = world / len(nodes)
lrank = int(rank % ngpus)
os.environ["CUDA_VISIBLE_DEVICES"] = str(lrank)

print(nodes)
id = int([x.split(".")[0] for x in nodes].index(os.uname()[1]))

print(f"Hello from process {rank}")
print({"id": id, "master": nodes[0], "nodes": len(nodes), "world": world})

jax.distributed.initialize(
    coordinator_address=f"{nodes[0]}:29500",
    num_processes=world,  # len(nodes),
    process_id=rank,  # id,
    # cluster_detection_method='mpi4py'
)


def show_shardings(shards: dict[str, NamedSharding]):
    for k, shard in shards.items():
        print(f"Mesh: {k} | {shard}")
        x = jax.random.normal(
            jax.random.PRNGKey(42), (jax.device_count(), jax.device_count())
        )
        _x = jax.device_put(x, shard)
        vas(_x)


print(jax.devices())
mesh = Mesh(
    devices=np.array(jax.devices()).reshape(-1, 1),
    axis_names=("data", "model"),
)
print(mesh)

shards = {
    "dp": NamedSharding(mesh, PartitionSpec("data", None)),
    "rep": NamedSharding(mesh, PartitionSpec()),
}
show_shardings(shards)

print("done")


jax.distributed.shutdown()
