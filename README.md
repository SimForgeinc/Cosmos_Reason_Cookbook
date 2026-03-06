# Customizing Intelligent Transportation Post-Training with Cosmos Reason 2
This code contains the files necessary for running post-training with cosmos reason 2 after using the Synthetic Data generation (SDG) for Traffic Scenarios. Everything here works under the assumption that Synthetic data generation has already occured.  For more information go to:
https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/reason2/intelligent-transportation/post_training.html

## Quick Start

### 1. Data Preparation
Create a question/answer pair json file similar to this structure:
``` qa.json
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
### 2. Training

### 3. Evaluate Results
Run the following:
```
python3 post_training_inference.py --model /path/to/saved/model/with/safetensors --video /path/to/output.mp4
```
