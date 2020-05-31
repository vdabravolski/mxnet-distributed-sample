from argparse import ArgumentParser
import logging
import subprocess
import os
import json
from contextlib import contextmanager
import signal
import time
import socket
import sys

fmt = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(format=fmt, level=logging.INFO)


def common_setup():

    # Read info that SageMaker provides
    current_host = os.environ['SM_CURRENT_HOST']
    hosts = json.loads(os.environ['SM_HOSTS'])

    # Enable SSH connections between containers
    _start_ssh_daemon()

    if current_host == _get_master_host_name(hosts):
        _wait_for_worker_nodes_to_start_sshd(hosts)

def _start_ssh_daemon():
    subprocess.Popen(["/usr/sbin/sshd", "-D"])

def _get_master_host_name(hosts):
    return sorted(hosts)[0]

def _wait_for_worker_nodes_to_start_sshd(hosts, interval=1, timeout_in_seconds=180):
    with timeout(seconds=timeout_in_seconds):
        while hosts:
            print("hosts that aren't SSHable yet: %s", str(hosts))
            for host in hosts:
                ssh_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                if _can_connect(host, 22, ssh_socket):
                    hosts.remove(host)
            time.sleep(interval)

class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds=0, minutes=0, hours=0):
    """
    Add a signal-based timeout to any block of code.
    If multiple time units are specified, they will be added together to determine time limit.
    Usage:
    with timeout(seconds=5):
        my_slow_function(...)
    Args:
        - seconds: The time limit, in seconds.
        - minutes: The time limit, in minutes.
        - hours: The time limit, in hours.
    """

    limit = seconds + 60 * minutes + 3600 * hours

    def handler(signum, frame):  # pylint: disable=W0613
        raise TimeoutError('timed out after {} seconds'.format(limit))

    try:
        signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, limit)
        yield
    finally:
        signal.alarm(0)

def _can_connect(host, port, s):
    try:
        print("testing connection to host %s", host)
        s.connect((host, port))
        s.close()
        print("can connect to host %s", host)
        return True
    except socket.error:
        print("can't connect to host %s", host)
        return False

    

def get_training_world(local=False):

    """
    Calculates number of devices in Sagemaker distributed cluster
    """
    
    # Get params of Sagemaker distributed cluster from predefined env variables
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    num_cpus = int(os.environ["SM_NUM_CPUS"])
    hosts = json.loads(os.environ["SM_HOSTS"])
    current_host = os.environ["SM_CURRENT_HOST"]
    
    # Define PyTorch training world
    world = {}
    world["number_of_processes"] = num_gpus if num_gpus > 0 else num_cpus
    world["number_of_machines"] = len(hosts)
    world["size"] = world["number_of_processes"] * world["number_of_machines"]
    world["machine_rank"] = hosts.index(current_host)
    world["current_host"] = current_host
    world["is_master"] = current_host == sorted(hosts)[0]
    
    if local:
        world["hosts"] = "localhost:{}".format(world["number_of_processes"])
        return world

    stitched_hosts = "" # prepare hosts line used in MPI run
    for host in hosts:
        stitched_hosts += "{}:{},".format(host, world["number_of_processes"])
        world["hosts"] = stitched_hosts[:-1]
    
    return world


def worker_routine(proccess_id_string, worker):
    """
    This method waits is executed on worker side. 
    It waits for 60 seconds and then checks if training processes are spawned up.
    """
    
    print("Inside worker routine")

    training_process_started = False
    
    while True:
        time.sleep(60)
                
        training_process_ps = subprocess.check_output('ps -elf | grep "{}"'.format(proccess_id_string), encoding='utf-8', shell=True)
        training_process_count = subprocess.check_output('ps -elf | grep "{}" | wc -l'.format(proccess_id_string), encoding='utf-8', shell=True)
        training_process_count_str = training_process_count.replace("\n", "").strip()
        training_process_count = int(training_process_count_str) - 2
        training_process_running = training_process_count > 0
        if training_process_started:
            print('training processes running: {}'.format(training_process_count))
            if not training_process_running:
                print('Worker {} training completed.'.format(worker))
                time.sleep(5)
                sys.exit(0)

        if not training_process_started:
            if training_process_running:
                training_process_started = True
            else:
                print('Worker {} exiting: training not started in 60 seconds.'.format(worker))
                sys.exit(1)


def master_routine(world, train_script, train_args):
    
    if_name = os.environ['SM_NETWORK_INTERFACE_NAME']
    
    # MPI run  config, according to HVD documentation: https://horovod.readthedocs.io/en/stable/mpirun.html
    mpi_cmd =  ("mpirun -np {} -H {} ".format(world["size"], world["hosts"]),
              "--allow-run-as-root "
              "--display-map "
              "--tag-output "
              "-mca btl_tcp_if_include {} ".format(if_name),
              "-mca oob_tcp_if_include {} ".format(if_name),
              "-x NCCL_SOCKET_IFNAME={} ".format(if_name),
              "-mca plm_rsh_no_tree_spawn 1 "
              "--bind-to none "
              "--map-by slot "
              "-mca orte_abort_on_non_zero_status 1 "
              "-x NCCL_DEBUG=INFO "
              "-x LD_LIBRARY_PATH "
              "-x PATH "
              "-mca pml ob1 "
              "-mca btl ^openib ")

    mpi_cmd = ''.join(mpi_cmd)
    
    
    # Train script config
    train_cmd = " python {} {}".format(args.train_script, train_args)    
    
    # Concat MPI run configuration and training script and its parameters
    joint_cmd = mpi_cmd+train_cmd
    print("*********joint_cmd******* \n",joint_cmd)
    
    process = subprocess.Popen(joint_cmd,  stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True) # TODO: consider refactoring to avoid shell=True
    
    while True:
        output = process.stdout.readline()
        
        if process.poll() is not None:
            break
        if output:
            print(output.decode("utf-8").strip())
    rc = process.poll()
    
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=joint_cmd)
    
    sys.exit(process.returncode)


if __name__ == "__main__":

    # Start SSH daemon across all nodes and establish communication
    common_setup()

    # Parse common arguments
    print('Starting training...')
    parser = ArgumentParser()
    parser.add_argument('--train-script', type=str, required=True, help="specify training script to run")
    parser.add_argument('--local', type=str, default="false", help="specify if you want to run locally")
    args, unknown = parser.parse_known_args()
    train_args = ' '.join(unknown)

    #is_local is a workaround for scenarios when we are running training job on local host only.     
    is_local = True if args.local.lower()=="true" else False
    # world captures parameters of Sagemaker training cluster
    world = get_training_world(is_local)

    # Define role: master or worker
    is_master = world["is_master"] 

    # if it's worker, then wait for training process to start and complete
    if not is_master:
        print("Worker: ", world["current_host"])
        process_search_term = "python {}".format(args.train_script) # TODO: change it to python3.6 once we bump to new version
        worker_routine(process_search_term, world["current_host"])
        print("Worker {} has completed".format(world["current_host"]))

    # if it's master, then execute master routine
    print("Master node is {}".format(world["current_host"]))
    master_routine(world, args.train_script, train_args)

    

    