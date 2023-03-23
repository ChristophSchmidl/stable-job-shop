import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.logger import get_logger



if __name__ == "__main__":
    # 1. Start a W&B Run
    logger = get_logger()

    run = wandb.init(
        project="cat-classification",
        notes="My first experiment",
        group="experiment_1",
        job_type="example-job",
        tags=["baseline", "paper1"] # useful for rl vs cp vs supervised
    )

    # You can also use wandb run within a context manager, then you do not have
    # to invoke run.finish() at the end
    #with wandb.init(...) as run:
    #    do-stuff
    # end of the run that is tracked by wandb


    # Handy when using argparse
    # wandb.config.update(args) # adds all of the arguments as config variables
    wandb.config.update({
        "epochs": 100, 
        "learning_rate": 0.001, 
        "batch_size": 128
    })

    df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
    wandb.log({'my_dataframe': wandb.Table(dataframe=df)})
    wandb.log({"accuracy": 4, "loss": 2})

    arr = np.array([[1, 2], [3, 4], [5, 6]])
    np.savez_compressed('data.npz', arr)
    #np.load('data.npz')

    artifact = wandb.Artifact(
        name="my_numpy_array.npz",
        type="data",
        description="This is an example array",
        metadata = {
            "key 1": "value 1"
        }
    )

    artifact.add_file("data.npz")
    run.log_artifact(artifact)

    logger.info("Finishing run...")
    run.finish()