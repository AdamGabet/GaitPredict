import wandb
import os
from model.training.utils.training_helper import *

from model.training.train_loop import train
from model.training.eval_loop import run_final_evaluation
from dotenv import load_dotenv
load_dotenv(interpolate=True)


def load_and_resume(run_id=None, project="GaitPredict", entity="adam-gabet-weizmann-institute-of-science"):
    """
    Load config from a previous W&B run and either:
    - Start a NEW run with the same config
    - Resume the SAME run if run_id is given
    """

    if run_id:
        # Resume the same run
        print(f"Resuming run {run_id}...")
        run = wandb.init(
            project=project,
            entity=entity,
            id=run_id,
            resume="must"  # 'must' = must resume this run_id, error if missing
        )
        config = run.config
    return run, config

def keep_training():
    # Example usage:
    run_id = "37zxande"
    run, config = load_and_resume(run_id=run_id)
    model_dir = get_latest_dir(find_model_dir_for_run(run_id, os.getenv("MODEL_SAVE_DIR")))
    run.config.update({"training_type": 'continue'}, allow_val_change=True)
    run.config.update({"model_file": model_dir}, allow_val_change=True)
    context = train(wandb.config, only_eval=True)

if __name__ == "__main__":
    keep_training()

