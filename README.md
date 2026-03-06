# Customizing Intelligent Transportation Post-Training with Cosmos Reason 2
This code contains the files necessary for running post-training with cosmos reason 2 after using the Synthetic Data generation (SDG) for Traffic Scenarios. Everything here works under the assumption that Synthetic data generation has already occured.  For more information go to:
https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/reason2/intelligent-transportation/post_training.html

## Quick Start

### 1. Data Preparation
Create a question/answer pair `qa.json` file with this structure:
``` json
[
  {
    "video_path": "path/to/video/example.mp4",
    "q": "What is happening in the video?",
    "a": "Description of what happens in the video."
  },
  {
    "video_path": "training_videos/rfs_evening_unprotected_left_with_emergency_vehicles.mp4",
    "q": "Is there a pedestrian visible?",
    "a": "No, pedestrians aren't visible."
  }
]
```

Then run the following:
```
python post_training_inference.py --qa qa.json --out training.json
```
The resulting `training.json` is the input into the next Supervised Fine Tuning stage. 

### 2. Post-Training with Supervised Fine Tuning (SFT)
Training follows the resources used in the link above, but for reference, this is the relevant information for the fine-tuning step
```
# 1. Make sure the environment is set up following Cosmos Reason 2 post-training setup guide: https://github.com/nvidia-cosmos/cosmos-reason2/blob/main/examples/cosmos_rl/README.md#setup

# 2. Activate the environment from cosmos-reason2/examples/cosmos_rl directory
source .venv/bin/activate
# 3. Run the training command from cosmos-cookbook/scripts/examples/reason2/intelligent-transportation directory
cosmos-rl --config sft_config.toml custom_sft.py
```
It is also worth mentioning the configuration parameters that need to be updated. The following serves as the example from the link above:
```
[custom.dataset]
annotation_path = "/path/to/training.json"
media_path = "/path/to/training/videos"
system_prompt = "You are a helpful assistant."

[custom.vision]
nframes = 8

[train]
optm_lr = 2e-5
output_dir = "outputs/output"
optm_warmup_steps = 0
optm_decay_type = "cosine"
optm_weight_decay = 0.01
train_batch_per_replica = 32
enable_validation = false
compile = false

[policy]
model_name_or_path = "nvidia/Cosmos-Reason2-2B"
model_max_length = 32768

[logging]
logger = ['console', 'wandb']
project_name = "cosmos_reason2"
experiment_name = "post_training/experiment_name"

[train.train_policy]
type = "sft"
mini_batch = 1
dataset.test_size = 0
dataloader_num_workers = 4
dataloader_prefetch_factor = 4

[train.ckpt]
enable_checkpoint = true
save_freq = 20

[policy.parallelism]
tp_size = 1
cp_size = 1
dp_shard_size = 8
pp_size = 1
```

### 3. Evaluate Results
Finally, after the model has been trained, in order to evaluate qualitative results, we run the following:
```
python3 post_training_inference.py --model /path/to/saved/model/with/safetensors --video /path/to/output.mp4
```
