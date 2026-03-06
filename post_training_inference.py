import argparse
import cv2
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq  # or AutoModelForImageTextToText

# -----------------------------
# Default settings
# -----------------------------
MAX_FRAMES = 8
MAX_NEW_TOKENS = 200
FRAME_SIZE = (384, 384)  # model expects 384x384
VIDEO_FPS = 24  # for metadata

# -----------------------------
# Load model
# -----------------------------
def load_model(model_dir):
    model_path = model_dir if Path(model_dir).exists() else model_dir
    print(f"Loading model from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto"
    )
    return processor, model

# -----------------------------
# Extract video frames as PIL images
# -----------------------------
def load_video_frames(video_path, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // max_frames, 1)
    frame_id = 0

    while cap.isOpened() and len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR -> RGB and resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, FRAME_SIZE)
        frames.append(Image.fromarray(frame))
        frame_id += step

    cap.release()
    print(f"Loaded {len(frames)} frames as PIL images")
    return frames

# -----------------------------
# Run inference
# -----------------------------
def run_inference(model_dir, video_path):
    processor, model = load_model(model_dir)
    frames_pil = load_video_frames(video_path, MAX_FRAMES)

    # --- Ensure placeholders match video tokens ---
    patches_per_frame = 36  # adjust if needed for your model
    total_video_tokens = len(frames_pil) * patches_per_frame

    # Generate prompt with exact number of placeholders
    user_prompt = "Describe what is happening in this driving scene: " + ("<|placeholder|>" * total_video_tokens)

    # Processor handles video internally
    inputs = processor(
        text=user_prompt,
        videos=[frames_pil],   # batch of 1 video
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS
        )

    text_output = processor.decode(outputs[0], skip_special_tokens=True)
    print("\n=== Model Output ===\n")
    print(text_output)

    return frames_pil, text_output

# -----------------------------
# Entry point with argparse
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cosmos Reason video inference")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model or base model")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--frames", type=int, default=MAX_FRAMES, help="Number of frames to extract")
    parser.add_argument("--tokens", type=int, default=MAX_NEW_TOKENS, help="Max tokens to generate")
    args = parser.parse_args()

    # Override globals if needed
    MAX_FRAMES = args.frames
    MAX_NEW_TOKENS = args.tokens

    frame_list, text_output = run_inference(args.model, args.video)