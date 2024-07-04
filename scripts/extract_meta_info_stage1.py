# pylint: disable=R0801
"""
This module is used to extract meta information from video directories.

It takes in two command-line arguments: `root_path` and `dataset_name`. The `root_path`
specifies the path to the video directory, while the `dataset_name` specifies the name
of the dataset. The module then collects all the video folder paths, and for each video
folder, it checks if a mask path and a face embedding path exist. If they do, it appends
a dictionary containing the image path, mask path, and face embedding path to a list.

Finally, the module writes the list of dictionaries to a JSON file with the filename
constructed using the `dataset_name`.

Usage:
    python tools/extract_meta_info_stage1.py --root_path /path/to/video_dir --dataset_name hdtf

"""

import argparse
import json
import os
from pathlib import Path

import torch


def collect_video_folder_paths(root_path: Path) -> list:
    """
    Collect all video folder paths from the root path.

    Args:
        root_path (Path): The root directory containing video folders.

    Returns:
        list: List of video folder paths.
    """
    return [frames_dir.resolve() for frames_dir in root_path.iterdir() if frames_dir.is_dir()]


def construct_meta_info(frames_dir_path: Path) -> dict:
    """
    Construct meta information for a given frames directory.

    Args:
        frames_dir_path (Path): The path to the frames directory.

    Returns:
        dict: A dictionary containing the meta information for the frames directory, or None if the required files do not exist.
    """
    mask_path = str(frames_dir_path).replace("images", "face_mask") + ".png"
    face_emb_path = str(frames_dir_path).replace("images", "face_emb") + ".pt"

    if not os.path.exists(mask_path):
        print(f"Mask path not found: {mask_path}")
        return None

    if torch.load(face_emb_path) is None:
        print(f"Face emb is None: {face_emb_path}")
        return None

    return {
        "image_path": str(frames_dir_path),
        "mask_path": mask_path,
        "face_emb": face_emb_path,
    }


def main():
    """
    Main function to extract meta info for training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root_path", type=str,
                        required=True, help="Root path of the video directories")
    parser.add_argument("-n", "--dataset_name", type=str,
                        required=True, help="Name of the dataset")
    parser.add_argument("--meta_info_name", type=str,
                        help="Name of the meta information file")

    args = parser.parse_args()

    if args.meta_info_name is None:
        args.meta_info_name = args.dataset_name

    image_dir = Path(args.root_path) / "images"
    output_dir = Path("./data")
    output_dir.mkdir(exist_ok=True)

    # Collect all video folder paths
    frames_dir_paths = collect_video_folder_paths(image_dir)

    meta_infos = []
    for frames_dir_path in frames_dir_paths:
        meta_info = construct_meta_info(frames_dir_path)
        if meta_info:
            meta_infos.append(meta_info)

    output_file = output_dir / f"{args.meta_info_name}_stage1.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(meta_infos, f, indent=4)

    print(f"Final data count: {len(meta_infos)}")


if __name__ == "__main__":
    main()
