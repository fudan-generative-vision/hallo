# pylint: disable=R0801
"""
This module is used to extract meta information from video files and store them in a JSON file.

The script takes in command line arguments to specify the root path of the video files,
the dataset name, and the name of the meta information file. It then generates a list of
dictionaries containing the meta information for each video file and writes it to a JSON
file with the specified name.

The meta information includes the path to the video file, the mask path, the face mask
path, the face mask union path, the face mask gaussian path, the lip mask path, the lip
mask union path, the lip mask gaussian path, the separate mask border, the separate mask
face, the separate mask lip, the face embedding path, the audio path, the vocals embedding
base last path, the vocals embedding base all path, the vocals embedding base average
path, the vocals embedding large last path, the vocals embedding large all path, and the
vocals embedding large average path.

The script checks if the mask path exists before adding the information to the list.

Usage:
    python tools/extract_meta_info_stage2.py --root_path <root_path> --dataset_name <dataset_name> --meta_info_name <meta_info_name>

Example:
    python tools/extract_meta_info_stage2.py --root_path data/videos_25fps --dataset_name my_dataset --meta_info_name my_meta_info
"""

import argparse
import json
import os
from pathlib import Path

import torch
from decord import VideoReader, cpu
from tqdm import tqdm


def get_video_paths(root_path: Path, extensions: list) -> list:
    """
    Get a list of video paths from the root path with the specified extensions.

    Args:
        root_path (Path): The root directory containing video files.
        extensions (list): List of file extensions to include.

    Returns:
        list: List of video file paths.
    """
    return [str(path.resolve()) for path in root_path.iterdir() if path.suffix in extensions]


def file_exists(file_path: str) -> bool:
    """
    Check if a file exists.

    Args:
        file_path (str): The path to the file.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    return os.path.exists(file_path)


def construct_paths(video_path: str, base_dir: str, new_dir: str, new_ext: str) -> str:
    """
    Construct a new path by replacing the base directory and extension in the original path.

    Args:
        video_path (str): The original video path.
        base_dir (str): The base directory to be replaced.
        new_dir (str): The new directory to replace the base directory.
        new_ext (str): The new file extension.

    Returns:
        str: The constructed path.
    """
    return str(video_path).replace(base_dir, new_dir).replace(".mp4", new_ext)


def extract_meta_info(video_path: str) -> dict:
    """
    Extract meta information for a given video file.

    Args:
        video_path (str): The path to the video file.

    Returns:
        dict: A dictionary containing the meta information for the video.
    """
    mask_path = construct_paths(
        video_path, "videos", "face_mask", ".png")
    sep_mask_border = construct_paths(
        video_path, "videos", "sep_pose_mask", ".png")
    sep_mask_face = construct_paths(
        video_path, "videos", "sep_face_mask", ".png")
    sep_mask_lip = construct_paths(
        video_path, "videos", "sep_lip_mask", ".png")
    face_emb_path = construct_paths(
        video_path, "videos", "face_emb", ".pt")
    audio_path = construct_paths(video_path, "videos", "audios", ".wav")
    vocal_emb_base_all = construct_paths(
        video_path, "videos", "audio_emb", ".pt")

    assert_flag = True

    if not file_exists(mask_path):
        print(f"Mask path not found: {mask_path}")
        assert_flag = False
    if not file_exists(sep_mask_border):
        print(f"Separate mask border not found: {sep_mask_border}")
        assert_flag = False
    if not file_exists(sep_mask_face):
        print(f"Separate mask face not found: {sep_mask_face}")
        assert_flag = False
    if not file_exists(sep_mask_lip):
        print(f"Separate mask lip not found: {sep_mask_lip}")
        assert_flag = False
    if not file_exists(face_emb_path):
        print(f"Face embedding path not found: {face_emb_path}")
        assert_flag = False
    if not file_exists(audio_path):
        print(f"Audio path not found: {audio_path}")
        assert_flag = False
    if not file_exists(vocal_emb_base_all):
        print(f"Vocal embedding base all not found: {vocal_emb_base_all}")
        assert_flag = False

    video_frames = VideoReader(video_path, ctx=cpu(0))
    audio_emb = torch.load(vocal_emb_base_all)
    if abs(len(video_frames) - audio_emb.shape[0]) > 3:
        print(f"Frame count mismatch for video: {video_path}")
        assert_flag = False

    face_emb = torch.load(face_emb_path)
    if face_emb is None:
        print(f"Face embedding is None for video: {video_path}")
        assert_flag = False

    del video_frames, audio_emb

    if assert_flag:
        return {
            "video_path": str(video_path),
            "mask_path": mask_path,
            "sep_mask_border": sep_mask_border,
            "sep_mask_face": sep_mask_face,
            "sep_mask_lip": sep_mask_lip,
            "face_emb_path": face_emb_path,
            "audio_path": audio_path,
            "vocals_emb_base_all": vocal_emb_base_all,
        }
    return None


def main():
    """
    Main function to extract meta info for training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root_path", type=str,
                        required=True, help="Root path of the video files")
    parser.add_argument("-n", "--dataset_name", type=str,
                        required=True, help="Name of the dataset")
    parser.add_argument("--meta_info_name", type=str,
                        help="Name of the meta information file")

    args = parser.parse_args()

    if args.meta_info_name is None:
        args.meta_info_name = args.dataset_name

    video_dir = Path(args.root_path) / "videos"
    video_paths = get_video_paths(video_dir, [".mp4"])

    meta_infos = []

    for video_path in tqdm(video_paths, desc="Extracting meta info"):
        meta_info = extract_meta_info(video_path)
        if meta_info:
            meta_infos.append(meta_info)

    print(f"Final data count: {len(meta_infos)}")

    output_file = Path(f"./data/{args.meta_info_name}_stage2.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(meta_infos, f, indent=4)


if __name__ == "__main__":
    main()
