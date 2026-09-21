
import json
import os
import h5py
import argparse
import pandas as pd
import torch
import subprocess
import cv2

from pathlib import Path
from safetensors.torch import save_file
from tqdm import tqdm

def flatten_dict(d, parent_key="", sep="/"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def is_av1(file_path):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "csv=p=0",
                str(file_path)
            ],
            capture_output=True,
            text=True
        )
        return result.stdout.strip() == "av1"
    except:
        return False

def convert_to_h264(input_path, output_path):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-an",
            str(output_path)
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def extract_first_frame(video_path, save_path):
    tmp_video = None
    read_video = video_path
    if is_av1(video_path):
        tmp_video = "/tmp/temp_h264_video.mp4"
        convert_to_h264(video_path, tmp_video)
        read_video = tmp_video
    cap = cv2.VideoCapture(str(read_video))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read video: {video_path}")
    cv2.imwrite(str(save_path), frame)
    if (tmp_video is not None and os.path.exists(tmp_video)):
        os.remove(tmp_video)

def main(args):
    source_dir = Path(args.source_dir)
    source_dataset_dir = source_dir / args.dataset_name
    source_data_dir = source_dataset_dir / "data" / "chunk-000"
    source_meta_dir = source_dataset_dir / "meta"
    source_videos_dir = source_dataset_dir / "videos" / "chunk-000"
    target_dir = Path(args.target_dir)
    target_images_dir = target_dir / "images" / args.dataset_name
    target_transitions_dir = target_dir / "transitions" / args.dataset_name
    target_meta_dir = target_transitions_dir / "meta_data"
    target_images_dir.mkdir(parents=True, exist_ok=True)
    target_transitions_dir.mkdir(parents=True, exist_ok=True)
    target_meta_dir.mkdir(parents=True, exist_ok=True)
    csv_file = target_dir / f"{args.dataset_name}.csv"
    COLUMNS = [
        "videoid",
        "contentUrl",
        "duration",
        "data_dir",
        "instruction",
        "dynamic_confidence",
        "dynamic_wording",
        "dynamic_source_category",
        "embodiment",
        "fps"
    ]
    df = pd.DataFrame(columns=COLUMNS)
    with open(source_meta_dir / "info.json", "r") as f:
        info = json.load(f)
    total_episodes = (info["total_episodes"])
    with open(source_meta_dir / "tasks.jsonl", "r") as f:
        tasks = [json.loads(line) for line in f]
    instruction = (tasks[0]["task"])
    source_video_views = [x for x in source_videos_dir.iterdir() if x.is_dir()]
    print("camera views:", [x.name for x in source_video_views])
    all_actions = []
    all_states = []
    for idx in tqdm(range(total_episodes)):
        image_paths = []
        for source_view_dir in source_video_views:
            view_name = source_view_dir.name
            target_camera_dir = target_images_dir / view_name
            target_camera_dir.mkdir(parents=True, exist_ok=True)
            source_video = source_view_dir / f"episode_{idx:06d}.mp4"
            image_path = target_camera_dir / f"{idx}.png"
            extract_first_frame(source_video, image_path)
            image_paths.append(str(Path(args.dataset_name) / view_name / f"{idx}.png"))
        episode_parquet_file = source_data_dir / f"episode_{idx:06d}.parquet"
        episode_data = pd.read_parquet(episode_parquet_file)
        actions = torch.tensor(episode_data["action"].tolist())
        states = torch.tensor(episode_data["observation.state"].tolist())
        target_h_file = target_transitions_dir / f"{idx}.h"
        with h5py.File(target_h_file, "w") as h5f:
            h5f.create_dataset("observation.state", data=states)
            h5f.create_dataset("action", data=actions)
            h5f.attrs["action_type"] = ("joint position")
            h5f.attrs["state_type"] = ("joint position")
            h5f.attrs["robot_type"] = (args.robot_name)
        all_actions.append(actions)
        all_states.append(states)
        df.loc[len(df)] = [
            idx,
            json.dumps(image_paths),
            "x",
            args.dataset_name,
            instruction,
            "x",
            "x",
            "x",
            args.robot_name,
            30
        ]
    actions = torch.cat(all_actions, dim=0)
    states = torch.cat(all_states, dim=0)
    stats = {
        "action": {
            "max": actions.max(dim=0).values,
            "min": actions.min(dim=0).values,
            "mean": actions.mean(dim=0),
            "std": actions.std(dim=0)
        },
        "observation.state": {
            "max": states.max(dim=0).values,
            "min": states.min(dim=0).values,
            "mean": states.mean(dim=0),
            "std": states.std(dim=0)
        }
    }
    save_file(flatten_dict(stats), target_meta_dir / "stats.safetensors")
    df.to_csv(csv_file, index=False)
    print(f">>> Finished create {args.dataset_name}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="./data"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True
    )
    parser.add_argument(
        "--robot_name",
        type=str,
        required=True
    )
    main(
        parser.parse_args()
    )


