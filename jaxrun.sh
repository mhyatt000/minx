# Enable GPU-MPI (if supported by application)
export MPICH_GPU_SUPPORT_ENABLED=1

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
# NRANKS_PER_NODE=8
NRANKS_PER_NODE=$(nvidia-smi -L | wc -l)
NRANKS=$(( NNODES * NRANKS_PER_NODE ))

NDEPTH=8
NTHREADS=1

cat $PBS_NODEFILE

echo
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE}"


echo "Job ID: $PBS_JOBID"
echo "Job Name: $PBS_JOBNAME"
echo "Node File: $PBS_NODEFILE"
echo "Number of Nodes: $PBS_NUM_NODES"
echo "Total Processors: $PBS_NP"
echo "Queue: $PBS_QUEUE"
echo "Working Directory: $PBS_O_WORKDIR"
echo "Submitting Host: $PBS_O_HOST"

module use /soft/modulefiles/ > /dev/null 2>&1
# module load conda
# source .venv/bin/activate 
# conda activate # newjax # bafl
# module load cudatoolkit-standalone/12.6.1
module load cudnn/9.4.0 > /dev/null 2>&1

#
# try debug nccl
#
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072

# export NCCL_SOCKET_IFNAME=eth0 # Adjust based on your interface name (e.g., eth0, ib0)
export NCCL_SOCKET_IFNAME=hsn # Adjust based on your interface name (e.g., eth0, ib0)
# export NCCL_DEBUG=TRACE          # Provides additional information if issues persist

echo $NCCL_SOCKET_IFNAME

#
# no proxy for dist jax
#

unset HTTP_PROXY
unset HTTPS_PROXY
unset https_proxy 
unset http_proxy

echo $HTTPS_PROXY
echo $HTTP_PROXY
echo

export TF_CPP_MIN_LOG_LEVEL=3 # silence annoying msg

export JAX_TRACEBACK_FILTERING=off
echo "about to run"
   
source .venv/bin/activate


# For applications that internally handle binding MPI/OpenMP processes to GPUs
# mpiexec -n ${NRANKS} --npernode ${NRANKS_PER_NODE} \
mpiexec -n ${NRANKS} --ppn ${NRANKS_PER_NODE} \
    ~/affinity_soph.sh python main.py
# ~/affinity_soph.sh python mvp.py

#
# sandbox
#

# ./run  

# ./unset_proxy_jax.sh  <args>

# mpiexec -np $NTORANKS --hostfile $PBS_NODEFILE  uv run main.py
# --envlist JAX_NUM_PROCESSES,JAX_COORDINATION_SERVICE_ADDR \
# --bind-to none \

# --envlist JAX_NUM_PROCESSES,JAX_COORDINATION_SERVICE_ADDR \
# --bind-to none \



# For applications that need mpiexec to bind MPI ranks to GPUs
#mpiexec -n ${NRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} -env OMP_PLACES=threads ./set_affinity_gpu_polaris.sh ./hello_affinity
