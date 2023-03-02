import time

import psutil
from gym.envs.registration import register

from safepo.common.runner import Runner

ENVS = {
    'storage_constrained_tuning_env':
        'robust_constrained_tuning.tuning_env:StorageConstrainedTuningEnv',
}


def train_eval(config):
    algo = config['algo']
    use_mpi = config['use_mpi']
    epoch = config['epoch']
    env = config['env']
    env_config = config['env_config']
    env_id = "%s_%smb-v0" % (env_config['benchmark'], env_config['storage_constraint'])

    env_class = ENVS[env]
    register(
        id=env_id,
        entry_point=env_class,
        max_episode_steps=100,
    )

    physical_cores = psutil.cpu_count(logical=False)  # exclude hyper-threading
    # Seed must be < 2**32 => use 2**16 to allow seed += 10000*proc_id() for MPI
    random_seed = int(time.time()) % 2 ** 16
    default_log_dir = "./Robust_%s_Constrained_%s" % (env_config['robust'], env_config['constrained'])

    if env_config['storage_constraint'] == 10000:
        unparsed_args = ["--pi_lr", "2e-4"]
    else:
        unparsed_args = []
    model = Runner(
        algo=algo,
        env_id=env_id,
        log_dir=default_log_dir,
        env_cfg=env_config,
        init_seed=random_seed,
        unparsed_args=unparsed_args,
        use_mpi=use_mpi
    )

    model.compile(num_runs=1, num_cores=8)
    model.train(epoch)
    model.eval()


def experiments_with_different_cost_constraints():
    config = {
        'algo': 'ppo',
        'use_mpi': True,
        'epoch': 20,
        'env': 'storage_constrained_tuning_env',
        'env_config': {
            'benchmark': 'tpcds',
            'db_name': 'tpcds_10',
            'robust': True,
            'constrained': True,
            'sample_as_proxy': True,
            'sample_file': None,
            "storage_constraint": 5000,
            "cost_limit": 1.20361852,
            "uncertainty_bound": 0.05,
        },
    }

    # storage budget 5000MB
    for i in [0.5, 0.6, 0.7, 0.8]:
        maintenance_cost = 2.4072370326072616
        config['env_config']['storage_constraint'] = 5000
        config['env_config']['cost_limit'] = maintenance_cost * i
        train_eval(config)

    # storage budget 7500MB
    for i in [0.5, 0.6, 0.7, 0.8]:
        maintenance_cost = 3.347175139322912
        config['env_config']['storage_constraint'] = 7500
        config['env_config']['cost_limit'] = maintenance_cost * i
        train_eval(config)

    # storage budget 7500MB
    for i in [0.5, 0.6, 0.7, 0.8]:
        maintenance_cost = 4.034503567096869
        config['env_config']['storage_constraint'] = 10000
        config['env_config']['cost_limit'] = maintenance_cost * i
        train_eval(config)


def experiments_with_different_uncertainty_bound():
    config = {
        'algo': 'ppo',
        'use_mpi': True,
        'epoch': 20,
        'env': 'storage_constrained_tuning_env',
        'env_config': {
            'benchmark': 'tpcds',
            'db_name': 'tpcds_10',
            'robust': True,
            'constrained': True,
            'sample_as_proxy': True,
            'sample_file': None,
            "storage_constraint": 5000,
            "cost_limit": 1.20361852,
            "uncertainty_bound": 0.05,
        },
    }

    maintenance_cost = 2.4072370326072616
    config['env_config']['storage_constraint'] = 5000
    config['env_config']['cost_limit'] = maintenance_cost
    for b in [0.03, 0.04, 0.06]:
        train_eval(config)
        config['env_config']['uncertainty_bound'] = b
        train_eval(config)


def experiments_sampled_based():
    config = {
        'algo': 'ppo',
        'use_mpi': True,
        'epoch': 20,
        'env': 'storage_constrained_tuning_env',
        'env_config': {
            'benchmark': 'tpcds',
            'db_name': 'tpcds_10',
            'robust': True,
            'constrained': True,
            'sample_as_proxy': True,
            'sample_file': None,
            "storage_constraint": 5000,
            "cost_limit": 1.20361852,
            "uncertainty_bound": 0.05,
        },
    }

    maintenance_cost = 2.4072370326072616
    config['env_config']['storage_constraint'] = 5000
    config['env_config']['cost_limit'] = maintenance_cost
    for i in [1, 2, 3, 4, 5]:
        sample_file = "sample%s.npy" % i
        config['env_config']['sample_file'] = sample_file
        train_eval(config)


if __name__ == '__main__':
    # experiments_with_different_cost_constraints()
    # experiments_with_different_uncertainty_bound()
    experiments_sampled_based()
