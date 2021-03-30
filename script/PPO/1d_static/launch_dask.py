from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import itertools
from stablebaseline_ppo import main_exp


###  from collections import namedtuple  
###  should replace the parser


learning_rate = 0.00025
Cliprange = [0.1, 0.2]
N_steps = [1e4, 1e5, 1e6]
Nminibatches = [10, 100, 1000]



parallel_args = []
for (n_steps, nminibatches, cliprange) in itertools.product(N_steps, Nminibatches, Cliprange):
        args = {}
        args["tb_log_name"] = f"clip{cliprange}_nsetps_{n_steps}_nminibatches_{nminibatches}"
        args["gamma"] = 0.99
        args["n_steps"] = int(n_steps)
        args["noptepochs"] = 4
        args["ent_coef"] = 0.01
        args["learning_rate"] = learning_rate
        args["vf_coef"] = 0.5
        args["cliprange"] = cliprange
        args["nminibatches"] = int(nminibatches)
        parallel_args.append(args)

print(parallel_args)



env_extra = ['source activate rl']



if __name__=='__main__':
        cluster = SLURMCluster(job_extra=['--cpus-per-task=1', '--ntasks-per-node=1'],
                        cores=1, processes=1,
                        memory='16GB',
                        walltime='96:00:00',
                        interface='ib0',
                        log_directory='log_dask',
                        local_directory='log_dask')


        n_workers = 9
        cluster.scale(n_workers)
        client = Client(cluster)
        print(client.cluster)

        results = [client.submit(main_exp, args) for args in parallel_args]
        print(results)
        print(client.gather(results))

