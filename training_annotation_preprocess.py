import os
import json
import random
import argparse


def load_qa_file(path):
    """Load QA entries and validate format."""
    with open(path, "r") as f:
        data = json.load(f)

    for i, item in enumerate(data):
        if "q" not in item or "a" not in item or "video_path" not in item:
            raise ValueError(
                f"Entry {i} must contain 'video_path', 'q', and 'a'"
            )

    return data


def build_dataset(
    qa_pairs,
    output,
    fps=30,
    shuffle=True,
    relative=True,
):
    """Build Cosmos-compatible SFT dataset."""

    # ---------- LOAD EXISTING DATASET ----------
    if os.path.exists(output):
        with open(output, "r") as f:
            existing = json.load(f)

        existing_pairs = {
            (item["videos"][0]["path"], item["conversations"][0]["value"])
            for item in existing
        }

    else:
        existing = []
        existing_pairs = set()

    id_counter = len(existing) + 1
    new_samples = []

    # ---------- BUILD DATASET ----------
    for qa in qa_pairs:

        video = qa["video_path"]

        if not os.path.exists(video):
            print(f"WARNING: video not found -> {video}")

        path = os.path.relpath(video) if relative else os.path.abspath(video)

        question_text = "<video>\n" + qa["q"].strip()

        if (path, question_text) in existing_pairs:
            continue

        sample = {
            "id": str(id_counter),
            "videos": [
                {
                    "path": path,
                    "fps": fps
                }
            ],
            "conversations": [
                {"from": "human", "value": question_text},
                {"from": "gpt", "value": qa["a"].strip()}
            ]
        }

        new_samples.append(sample)
        existing_pairs.add((path, question_text))
        id_counter += 1

    # ---------- MERGE ----------
    combined = existing + new_samples

    if shuffle:
        random.shuffle(combined)

    # ---------- SAVE ----------
    with open(output, "w") as f:
        json.dump(combined, f, indent=2)

    print("\nDataset build complete")
    print(f"Added {len(new_samples)} new samples")
    print(f"Total dataset size: {len(combined)}")
    print(f"Saved to {output}\n")


# ---------------- CLI ----------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--qa",
        required=True,
        help="QA JSON file containing video_path, q, a"
    )

    parser.add_argument(
        "--out",
        default="training.json",
        help="Output dataset file"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video FPS metadata"
    )

    parser.add_argument(
        "--no_shuffle",
        action="store_true"
    )

    parser.add_argument(
        "--absolute",
        action="store_true"
    )

    args = parser.parse_args()

    qa_pairs = load_qa_file(args.qa)

    build_dataset(
        qa_pairs=qa_pairs,
        output=args.out,
        fps=args.fps,
        shuffle=not args.no_shuffle,
        relative=not args.absolute
    )