import os
import time
from tqdm import tqdm
from functools import partial
from typing import *

import jax
import jax.distributed
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf

# from big_vision import sharding
# from big_vision.pp import utils
from flax import linen as nn
from flax import nnx
from flax.nnx.transforms.deprecated import N
from jax.debug import visualize_array_sharding as vas
from jax.experimental import multihost_utils as mx
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from minx import comm
from minx.net import modules
from minx.net.modules import FeedForward

from minx.comm import dist_print as print


for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(var, None)

os.environ["NCCL_SOCKET_IFNAME"] = "hsn"


def show_shardings(shards: dict[str, NamedSharding]):
    for k, shard in shards.items():
        print(f"Mesh: {k} | {shard}")
        x = jax.random.normal(
            jax.random.PRNGKey(42), (jax.device_count(), jax.device_count())
        )
        _x = jax.device_put(x, shard)
        vas(_x)


def tmp2():

    print(jax.devices())
    print(type(jax.devices()))

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

    def create(cls, *args, **kwargs):

        @nnx.jit
        def create_sharded_model():
            # model = DotReluDot(1024, rngs=nnx.Rngs(0))  # Unsharded at this moment.
            _model = cls(*args, **kwargs)
            state = nnx.state(_model)  # The model's state, a pure pytree.
            pspecs = nnx.get_partition_spec(
                state
            )  # Strip out the annotations from state.
            sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
            nnx.update(_model, sharded_state)  # The model is sharded now!
            return _model

        with mesh:
            sharded_model = create_sharded_model()
        return sharded_model

    model = create(FeedForward, features=1024, hidden_dim=32, rngs=nnx.Rngs(0))

    # They are some `GSPMDSharding` now - not a single device!
    # print(model.dot1.kernel.value.sharding)
    # print(model.w2.value.sharding)

    #
    #
    #
    # Load a sharded model from a checkpoint
    #
    #
    #

    optimizer = nnx.Optimizer(model, optax.adam(1e-3))  # reference sharing

    @nnx.jit
    def train_step(model, optimizer, x, y):
        def loss_fn(model):
            yp = model(x)
            # yp = jax.lax.with_sharding_constraint(yp, shards["rep"])
            print(yp.shape)
            return jnp.mean((yp - y) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)

        return loss

    bs = 4096
    todp = lambda _x: jax.device_put(_x, shards["dp"])
    x = todp(jax.random.normal(jax.random.key(1), (bs, 1024)))
    y = todp(jax.random.normal(jax.random.key(2), (bs, 1024)))

    vas(x)
    print(x.sharding)

    nsteps = int(1e1)
    with mesh:
        for _ in tqdm(range(nsteps)):
            tic = time.time()
            loss = train_step(model, optimizer, x, y)
            print(loss)
            toc = time.time()
            print(f'time: {round(tic-toc,2)})


def tmp():

    def shard(batch, *, mesh):
        return mx.host_local_array_to_global_array(batch, mesh, PartitionSpec("batch"))

    def unshard(batch, *, mesh):
        return mx.global_array_to_host_local_array(batch, mesh, PartitionSpec("batch"))

    globalbatch = 8
    d = tf.data.Dataset.range(16).repeat(int(1e3))
    d = d.map(lambda _: tf.random.uniform([], minval=0, maxval=1))  # rand
    d = d.batch(16).batch(globalbatch, drop_remainder=True)
    d = d.shard(2, jax.process_index())

    # make it a jnp then shard
    d = map(
        lambda x: shard(jnp.array(x)),
        iter(d),
    )
    d = iter(d)

    # print(next(d))


def main():

    comm.init()
    print("initialized!")

    # mx.sync_global_devices()

    comm.show()

    tmp2()
    quit()

    """
    jax.distributed.initialize(
        coordinator_address="gpu1:29500",
        num_processes=os.environ.get("SLURM_NNODES"),
        local_device_ids=None,
        initialization_timeout=300,
    )
    print('done')
    """


if __name__ == "__main__":
    main()
