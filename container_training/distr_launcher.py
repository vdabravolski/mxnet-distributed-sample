import json
import os
import sys
from argparse import ArgumentParser
import logging
import signal

# import DMLC utils for distributed MPI run
# from dmlc_tracker import mpi
import mpi
from dmlc_tracker import opts

def create_hostfile(hosts):
    """
    MPI RTE requires to pass a hostfile, hence, creating a file.
    Returns:
      - filename
    """
    
    filename = "hostfile"
    
    with open(filename, mode='w') as file:
        file.write('\n'.join(hosts))
    
#     with open(filename, "w") as f:
#         print("hosts:", hosts)
#         print(type(hosts))
#         json.dump(hosts, f)
    
    return filename

def get_training_world():
    """
    Calculates number of devices in Sagemaker distributed cluster
    """
    
    # Get params of Sagemaker distributed cluster from predefined env variables
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    num_cpus = int(os.environ["SM_NUM_CPUS"])
    hosts = json.loads(os.environ["SM_HOSTS"])
    current_host = os.environ["SM_CURRENT_HOST"]
    
    # Define MXnet training world
    world = {}
    world["num_workers"] = num_gpus if num_gpus > 0 else num_cpus
    world["num_servers"] = len(hosts)
    world["hosts"] = hosts
    world["size"] = world["num_workers"] * world["num_servers"]
    world["machine_rank"] = hosts.index(current_host)
    
    return world

def dlmc_opts(world, command):
    """convert cluster parameters to DMLC's opts
    """
    
    parser = ArgumentParser(description="DLMC arguments")
    parser.add_argument("--num-workers", required=True, type=int, help='number of worker nodes to be launched')
    parser.add_argument("--num-servers", type=int, help='number of server nodes to be launched')
    parser.add_argument("--host-file", type=str, help = 'the hostfile of slave machines which will run')
    parser.add_argument('command', nargs='+', help='command for launching the program')

    args = ['--num-workers', str(world["num_workers"]),
            '--num-servers', str(world["num_servers"]),
            '--host-file', create_hostfile(world["hosts"])]
    
    args.extend([command])
    
    dlmc_args = parser.parse_args(args)
    
    return dlmc_args


def signal_handler(signal, frame):
    logging.info('Stop launcher')
    sys.exit(0)


if __name__ == "__main__":
    
#     TODO: remove it
#     import time
#     time.sleep(3600)
    
    fmt = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)
    signal.signal(signal.SIGINT, signal_handler)
    
    print('Starting training...')
    parser = ArgumentParser()
    parser.add_argument('--train-script', type=str, default="train_maskrcnn.py", help="specify training script to run")
#     parser.add_argument('command', nargs='+', help="command for launching training script")
    args, unknown = parser.parse_known_args()
    mpi_cmd = ' '.join(unknown)

    
    print(args)
    # Derive parameters of distributed training cluster in Sagemaker
    world = get_training_world()
    exec_cmd = "python {} {}".format(args.train_script, mpi_cmd)
    
    print("MPI run execution command: ", exec_cmd)
    args = dlmc_opts(world, exec_cmd)
    print("MPI final args: ", args)
    mpi.submit(args)