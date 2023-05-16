import wandb
import os
from src.logger import get_logger
from src.dispatching_rules import FIFO_worker, MWKR_worker, RANDOM_worker
import src.config as config
import pandas as pd
import numpy as np


def execute_fifo_worker(instance_paths):

    run = wandb.init(
        project=config.WANDB_PROJECT,
        notes="Dispatching rule: FIFO",
        group="dispatching-rules",
        job_type=f"FIFO",
        tags=["dispatching-rule", "baseline", "fifo"]    
    )

    wandb.config.update({ 
        "dispatching-rule": "FIFO"
    })

    instance_makespans = []

    for instance_path in instance_paths:
        instance_name = os.path.split(instance_path)[-1].split(sep=".")[0].upper()
        makespan = FIFO_worker(instance_name = instance_path)
        wandb.log({"make_span": makespan})
        instance_makespans.append(dict(
            Instance_name=instance_name,
            Makespan=makespan)
        )

    df = pd.DataFrame(instance_makespans)
    wandb.log({f"dispatching_rules_fifo_solutions": wandb.Table(dataframe=df)})
    run.finish()


def execute_mwkr_worker(instance_paths):
    run = wandb.init(
        project=config.WANDB_PROJECT,
        notes="Dispatching rule: MWKR",
        group="dispatching-rules",
        job_type=f"MWKR",
        tags=["dispatching-rule", "baseline", "mwkr"]    
    )

    wandb.config.update({ 
        "dispatching-rule": "MWKR"
    })

    
    instance_makespans = []

    for instance_path in instance_paths:
        instance_name = os.path.split(instance_path)[-1].split(sep=".")[0].upper()
        makespan = MWKR_worker(instance_name = instance_path)
        wandb.log({"make_span": makespan})
        instance_makespans.append(dict(
            Instance_name=instance_name,
            Makespan=makespan)
        )

    df = pd.DataFrame(instance_makespans)
    wandb.log({f"dispatching_rules_mwkr_solutions": wandb.Table(dataframe=df)})
    run.finish()

def execute_random_worker(instance_paths, n_runs=10):

    run = wandb.init(
        project=config.WANDB_PROJECT,
        notes="Dispatching rule: Random",
        group="dispatching-rules",
        job_type=f"random",
        tags=["dispatching-rule", "baseline", "random"]    
    )

    wandb.config.update({ 
        "dispatching-rule": "Random",
        "n_runs": n_runs
    })

    instance_mean_makespans = []

    for instance_path in instance_paths:
        instance_name = os.path.split(instance_path)[-1].split(sep=".")[0].upper()

        makespans = []
        for idx in range(n_runs):
            makespan = RANDOM_worker(instance_name = instance_path)
            makespans.append(makespan)
            wandb.log({"instance_name" : instance_name,"make_span": makespan})

        mean_makespan = np.mean(makespans)
        instance_mean_makespans.append(dict(
            Instance_name=instance_name,
            Mean_makespan=mean_makespan)
        )

    df = pd.DataFrame(instance_mean_makespans)
    wandb.log({f"dispatching_rules_random_solutions": wandb.Table(dataframe=df)})
    run.finish()


if __name__ == '__main__':
    pass
