import wandb
# Start a wandb run with `sync_tensorboard=True`

# Your training code using TensorBoard
wandb.tensorboard.patch(root_logdir="runs")

wandb.init(project='improvingroboticsds', sync_tensorboard=True)

# [Optional]Finish the wandb run to upload the tensorboard logs to W&B (if running in Notebook)
wandb.finish()
