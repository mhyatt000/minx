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

# from mpi4py import MPI

for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(var, None)

# os.environ["NCCL_SOCKET_IFNAME"] = "hsn"


import crossformer
from crossformer.model.crossformer_model import CrossFormerModel
from rich.pretty import pprint

model = CrossFormerModel.load_pretrained("hf://rail-berkeley/crossformer", None)

pprint(model.example_batch)

print("Model loaded")
