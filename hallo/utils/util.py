# pylint: disable=C0116
# pylint: disable=W0718
# pylint: disable=R1732
# pylint: disable=R0801
"""
utils.py

This module provides utility functions for various tasks such as setting random seeds,
importing modules from files, managing checkpoint files, and saving video files from 
sequences of PIL images.

Functions:
    seed_everything(seed)
    import_filename(filename)
    delete_additional_ckpt(base_path, num_keep)
    save_videos_from_pil(pil_images, path, fps=8)

Dependencies:
    importlib
    os
    os.path as osp
    random
    shutil
    sys
    pathlib.Path
    av
    cv2
    mediapipe as mp
    numpy as np
    torch
    torchvision
    einops.rearrange
    moviepy.editor.AudioFileClip, VideoClip
    PIL.Image

Examples:
    seed_everything(42)
    imported_module = import_filename('path/to/your/module.py')
    delete_additional_ckpt('path/to/checkpoints', 1)
    save_videos_from_pil(pil_images, 'output/video.mp4', fps=12)

The functions in this module ensure reproducibility of experiments by seeding random number 
generators, allow dynamic importing of modules, manage checkpoint files by deleting extra ones, 
and provide a way to save sequences of images as video files.

Function Details:
    seed_everything(seed)
        Seeds all random number generators to ensure reproducibility.

    import_filename(filename)
        Imports a module from a given file location.

    delete_additional_ckpt(base_path, num_keep)
        Deletes additional checkpoint files in the given directory.

    save_videos_from_pil(pil_images, path, fps=8)
        Saves a sequence of images as a video using the Pillow library.

Attributes:
    _ (str): Placeholder for static type checking
"""

import importlib
import os
import os.path as osp
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

import av
import cv2
import mediapipe as mp
import numpy as np
import torch
import torchvision
from einops import rearrange
from moviepy.editor import AudioFileClip, VideoClip
from PIL import Image


def seed_everything(seed):
    """
    Seeds all random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to set for all random number generators.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def import_filename(filename):
    """
    Import a module from a given file location.

    Args:
        filename (str): The path to the file containing the module to be imported.

    Returns:
        module: The imported module.

    Raises:
        ImportError: If the module cannot be imported.

    Example:
        >>> imported_module = import_filename('path/to/your/module.py')
    """
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def delete_additional_ckpt(base_path, num_keep):
    """
    Deletes additional checkpoint files in the given directory.

    Args:
        base_path (str): The path to the directory containing the checkpoint files.
        num_keep (int): The number of most recent checkpoint files to keep.

    Returns:
        None

    Raises:
        FileNotFoundError: If the base_path does not exist.

    Example:
        >>> delete_additional_ckpt('path/to/checkpoints', 1)
        # This will delete all but the most recent checkpoint file in 'path/to/checkpoints'.
    """
    dirs = []
    for d in os.listdir(base_path):
        if d.startswith("checkpoint-"):
            dirs.append(d)
    num_tot = len(dirs)
    if num_tot <= num_keep:
        return
    # ensure ckpt is sorted and delete the ealier!
    del_dirs = sorted(dirs, key=lambda x: int(
        x.split("-")[-1]))[: num_tot - num_keep]
    for d in del_dirs:
        path_to_dir = osp.join(base_path, d)
        if osp.exists(path_to_dir):
            shutil.rmtree(path_to_dir)


def save_videos_from_pil(pil_images, path, fps=8):
    """
    Save a sequence of images as a video using the Pillow library.

    Args:
        pil_images (List[PIL.Image]): A list of PIL.Image objects representing the frames of the video.
        path (str): The output file path for the video.
        fps (int, optional): The frames per second rate of the video. Defaults to 8.
    
    Returns:
        None
    
    Raises:
        ValueError: If the save format is not supported.

    This function takes a list of PIL.Image objects and saves them as a video file with a specified frame rate.
    The output file format is determined by the file extension of the provided path. Supported formats include
    .mp4, .avi, and .mkv. The function uses the Pillow library to handle the image processing and video
    creation.
    """
    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = pil_images[0].size

    if save_fmt == ".mp4":
        codec = "libx264"
        container = av.open(path, "w")
        stream = container.add_stream(codec, rate=fps)

        stream.width = width
        stream.height = height

        for pil_image in pil_images:
            # pil_image = Image.fromarray(image_arr).convert("RGB")
            av_frame = av.VideoFrame.from_image(pil_image)
            container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
        container.close()

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    """
    Save a grid of videos as an animation or video.

    Args:
        videos (torch.Tensor): A tensor of shape (batch_size, channels, time, height, width)
            containing the videos to save.
        path (str): The path to save the video grid. Supported formats are .mp4, .avi, and .gif.
        rescale (bool, optional): If True, rescale the video to the original resolution.
            Defaults to False.
        n_rows (int, optional): The number of rows in the video grid. Defaults to 6.
        fps (int, optional): The frame rate of the saved video. Defaults to 8.

    Raises:
        ValueError: If the video format is not supported.

    Returns:
        None
    """
    videos = rearrange(videos, "b c t h w -> t b c h w")
    # height, width = videos.shape[-2:]
    outputs = []

    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c h w)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)

        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_videos_from_pil(outputs, path, fps)


def read_frames(video_path):
    """
    Reads video frames from a given video file.

    Args:
        video_path (str): The path to the video file.

    Returns:
        container (av.container.InputContainer): The input container object
                                                   containing the video stream.

    Raises:
        FileNotFoundError: If the video file is not found.
        RuntimeError: If there is an error in reading the video stream.

    The function reads the video frames from the specified video file using the
    Python AV library (av). It returns an input container object that contains
    the video stream. If the video file is not found, it raises a FileNotFoundError,
    and if there is an error in reading the video stream, it raises a RuntimeError.
    """
    container = av.open(video_path)

    video_stream = next(s for s in container.streams if s.type == "video")
    frames = []
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            image = Image.frombytes(
                "RGB",
                (frame.width, frame.height),
                frame.to_rgb().to_ndarray(),
            )
            frames.append(image)

    return frames


def get_fps(video_path):
    """
    Get the frame rate (FPS) of a video file.

    Args:
        video_path (str): The path to the video file.

    Returns:
        int: The frame rate (FPS) of the video file.
    """
    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == "video")
    fps = video_stream.average_rate
    container.close()
    return fps


def tensor_to_video(tensor, output_video_file, audio_source, fps=25):
    """
    Converts a Tensor with shape [c, f, h, w] into a video and adds an audio track from the specified audio file.

    Args:
        tensor (Tensor): The Tensor to be converted, shaped [c, f, h, w].
        output_video_file (str): The file path where the output video will be saved.
        audio_source (str): The path to the audio file (WAV file) that contains the audio track to be added.
        fps (int): The frame rate of the output video. Default is 25 fps.
    """
    tensor = tensor.permute(1, 2, 3, 0).cpu(
    ).numpy()  # convert to [f, h, w, c]
    tensor = np.clip(tensor * 255, 0, 255).astype(
        np.uint8
    )  # to [0, 255]

    def make_frame(t):
        # get index
        frame_index = min(int(t * fps), tensor.shape[0] - 1)
        return tensor[frame_index]
    new_video_clip = VideoClip(make_frame, duration=tensor.shape[0] / fps)
    audio_clip = AudioFileClip(audio_source).subclip(0, tensor.shape[0] / fps)
    new_video_clip = new_video_clip.set_audio(audio_clip)
    new_video_clip.write_videofile(output_video_file, fps=fps, audio_codec='aac')


silhouette_ids = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]
lip_ids = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
           146, 91, 181, 84, 17, 314, 405, 321, 375]


def compute_face_landmarks(detection_result, h, w):
    """
    Compute face landmarks from a detection result.

    Args:
        detection_result (mediapipe.solutions.face_mesh.FaceMesh): The detection result containing face landmarks.
        h (int): The height of the video frame.
        w (int): The width of the video frame.

    Returns:
        face_landmarks_list (list): A list of face landmarks.
    """
    face_landmarks_list = detection_result.face_landmarks
    if len(face_landmarks_list) != 1:
        print("#face is invalid:", len(face_landmarks_list))
        return []
    return [[p.x * w, p.y * h] for p in face_landmarks_list[0]]


def get_landmark(file):
    """
    This function takes a file as input and returns the facial landmarks detected in the file.

    Args:
        file (str): The path to the file containing the video or image to be processed.

    Returns:
        Tuple[List[float], List[float]]: A tuple containing two lists of floats representing the x and y coordinates of the facial landmarks.
    """
    model_path = "pretrained_models/face_analysis/models/face_landmarker_v2_with_blendshapes.task"
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    # Create a face landmarker instance with the video mode:
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        image = mp.Image.create_from_file(str(file))
        height, width = image.height, image.width
        face_landmarker_result = landmarker.detect(image)
        face_landmark = compute_face_landmarks(
            face_landmarker_result, height, width)

    return np.array(face_landmark), height, width


def get_landmark_overframes(landmark_model, frames_path):
    """
    This function iterate frames and returns the facial landmarks detected in each frame.

    Args:
        landmark_model: mediapipe landmark model instance
        frames_path (str): The path to the video frames.

    Returns:
        List[List[float], float, float]: A List containing two lists of floats representing the x and y coordinates of the facial landmarks.
    """

    face_landmarks = []

    for file in sorted(os.listdir(frames_path)):
        image = mp.Image.create_from_file(os.path.join(frames_path, file))
        height, width = image.height, image.width
        landmarker_result = landmark_model.detect(image)
        frame_landmark = compute_face_landmarks(
            landmarker_result, height, width)
        face_landmarks.append(frame_landmark)

    return face_landmarks, height, width


def get_lip_mask(landmarks, height, width, out_path=None, expand_ratio=2.0):
    """
    Extracts the lip region from the given landmarks and saves it as an image.

    Parameters:
        landmarks (numpy.ndarray): Array of facial landmarks.
        height (int): Height of the output lip mask image.
        width (int): Width of the output lip mask image.
        out_path (pathlib.Path): Path to save the lip mask image.
        expand_ratio (float): Expand ratio of mask.
    """
    lip_landmarks = np.take(landmarks, lip_ids, 0)
    min_xy_lip = np.round(np.min(lip_landmarks, 0))
    max_xy_lip = np.round(np.max(lip_landmarks, 0))
    min_xy_lip[0], max_xy_lip[0], min_xy_lip[1], max_xy_lip[1] = expand_region(
        [min_xy_lip[0], max_xy_lip[0], min_xy_lip[1], max_xy_lip[1]], width, height, expand_ratio)
    lip_mask = np.zeros((height, width), dtype=np.uint8)
    lip_mask[round(min_xy_lip[1]):round(max_xy_lip[1]),
             round(min_xy_lip[0]):round(max_xy_lip[0])] = 255
    if out_path:
        cv2.imwrite(str(out_path), lip_mask)
        return None

    return lip_mask


def get_union_lip_mask(landmarks, height, width, expand_ratio=1):
    """
    Extracts the lip region from the given landmarks and saves it as an image.

    Parameters:
        landmarks (numpy.ndarray): Array of facial landmarks.
        height (int): Height of the output lip mask image.
        width (int): Width of the output lip mask image.
        expand_ratio (float): Expand ratio of mask.
    """
    lip_masks = []
    for landmark in landmarks:
        lip_masks.append(get_lip_mask(landmarks=landmark, height=height,
                     width=width, expand_ratio=expand_ratio))
    union_mask = get_union_mask(lip_masks)
    return union_mask


def get_face_mask(landmarks, height, width, out_path=None, expand_ratio=1.2):
    """
    Generate a face mask based on the given landmarks.

    Args:
        landmarks (numpy.ndarray): The landmarks of the face.
        height (int): The height of the output face mask image.
        width (int): The width of the output face mask image.
        out_path (pathlib.Path): The path to save the face mask image.
        expand_ratio (float): Expand ratio of mask.
    Returns:
        None. The face mask image is saved at the specified path.
    """
    face_landmarks = np.take(landmarks, silhouette_ids, 0)
    min_xy_face = np.round(np.min(face_landmarks, 0))
    max_xy_face = np.round(np.max(face_landmarks, 0))
    min_xy_face[0], max_xy_face[0], min_xy_face[1], max_xy_face[1] = expand_region(
        [min_xy_face[0], max_xy_face[0], min_xy_face[1], max_xy_face[1]], width, height, expand_ratio)
    face_mask = np.zeros((height, width), dtype=np.uint8)
    face_mask[round(min_xy_face[1]):round(max_xy_face[1]),
              round(min_xy_face[0]):round(max_xy_face[0])] = 255
    if out_path:
        cv2.imwrite(str(out_path), face_mask)
        return None

    return face_mask


def get_union_face_mask(landmarks, height, width, expand_ratio=1):
    """
    Generate a face mask based on the given landmarks.

    Args:
        landmarks (numpy.ndarray): The landmarks of the face.
        height (int): The height of the output face mask image.
        width (int): The width of the output face mask image.
        expand_ratio (float): Expand ratio of mask.
    Returns:
        None. The face mask image is saved at the specified path.
    """
    face_masks = []
    for landmark in landmarks:
        face_masks.append(get_face_mask(landmarks=landmark,height=height,width=width,expand_ratio=expand_ratio))
    union_mask = get_union_mask(face_masks)
    return union_mask

def get_mask(file, cache_dir, face_expand_raio):
    """
    Generate a face mask based on the given landmarks and save it to the specified cache directory.

    Args:
        file (str): The path to the file containing the landmarks.
        cache_dir (str): The directory to save the generated face mask.

    Returns:
        None
    """
    landmarks, height, width = get_landmark(file)
    file_name = os.path.basename(file).split(".")[0]
    get_lip_mask(landmarks, height, width, os.path.join(
        cache_dir, f"{file_name}_lip_mask.png"))
    get_face_mask(landmarks, height, width, os.path.join(
        cache_dir, f"{file_name}_face_mask.png"), face_expand_raio)
    get_blur_mask(os.path.join(
        cache_dir, f"{file_name}_face_mask.png"), os.path.join(
        cache_dir, f"{file_name}_face_mask_blur.png"), kernel_size=(51, 51))
    get_blur_mask(os.path.join(
        cache_dir, f"{file_name}_lip_mask.png"), os.path.join(
        cache_dir, f"{file_name}_sep_lip.png"), kernel_size=(31, 31))
    get_background_mask(os.path.join(
        cache_dir, f"{file_name}_face_mask_blur.png"), os.path.join(
        cache_dir, f"{file_name}_sep_background.png"))
    get_sep_face_mask(os.path.join(
        cache_dir, f"{file_name}_face_mask_blur.png"), os.path.join(
        cache_dir, f"{file_name}_sep_lip.png"), os.path.join(
        cache_dir, f"{file_name}_sep_face.png"))


def expand_region(region, image_w, image_h, expand_ratio=1.0):
    """
    Expand the given region by a specified ratio.
    Args:
        region (tuple): A tuple containing the coordinates (min_x, max_x, min_y, max_y) of the region.
        image_w (int): The width of the image.
        image_h (int): The height of the image.
        expand_ratio (float, optional): The ratio by which the region should be expanded. Defaults to 1.0.

    Returns:
        tuple: A tuple containing the expanded coordinates (min_x, max_x, min_y, max_y) of the region.
    """

    min_x, max_x, min_y, max_y = region
    mid_x = (max_x + min_x) // 2
    side_len_x = (max_x - min_x) * expand_ratio
    mid_y = (max_y + min_y) // 2
    side_len_y = (max_y - min_y) * expand_ratio
    min_x = mid_x - side_len_x // 2
    max_x = mid_x + side_len_x // 2
    min_y = mid_y - side_len_y // 2
    max_y = mid_y + side_len_y // 2
    if min_x < 0:
        max_x -= min_x
        min_x = 0
    if max_x > image_w:
        min_x -= max_x - image_w
        max_x = image_w
    if min_y < 0:
        max_y -= min_y
        min_y = 0
    if max_y > image_h:
        min_y -= max_y - image_h
        max_y = image_h

    return round(min_x), round(max_x), round(min_y), round(max_y)


def get_blur_mask(file_path, output_file_path, resize_dim=(64, 64), kernel_size=(101, 101)):
    """
    Read, resize, blur, normalize, and save an image.

    Parameters:
    file_path (str): Path to the input image file.
    output_dir (str): Path to the output directory to save blurred images.
    resize_dim (tuple): Dimensions to resize the images to.
    kernel_size (tuple): Size of the kernel to use for Gaussian blur.
    """
    # Read the mask image
    mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image is loaded successfully
    if mask is not None:
        normalized_mask = blur_mask(mask,resize_dim=resize_dim,kernel_size=kernel_size)
        # Save the normalized mask image
        cv2.imwrite(output_file_path, normalized_mask)
        return f"Processed, normalized, and saved: {output_file_path}"
    return f"Failed to load image: {file_path}"


def blur_mask(mask, resize_dim=(64, 64), kernel_size=(51, 51)):
    """
    Read, resize, blur, normalize, and save an image.

    Parameters:
    file_path (str): Path to the input image file.
    resize_dim (tuple): Dimensions to resize the images to.
    kernel_size (tuple): Size of the kernel to use for Gaussian blur.
    """
    # Check if the image is loaded successfully
    normalized_mask = None
    if mask is not None:
        # Resize the mask image
        resized_mask = cv2.resize(mask, resize_dim)
        # Apply Gaussian blur to the resized mask image
        blurred_mask = cv2.GaussianBlur(resized_mask, kernel_size, 0)
        # Normalize the blurred image
        normalized_mask = cv2.normalize(
            blurred_mask, None, 0, 255, cv2.NORM_MINMAX)
        # Save the normalized mask image
    return normalized_mask

def get_background_mask(file_path, output_file_path):
    """
    Read an image, invert its values, and save the result.

    Parameters:
    file_path (str): Path to the input image file.
    output_dir (str): Path to the output directory to save the inverted image.
    """
    # Read the image
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Failed to load image: {file_path}")
        return

    # Invert the image
    inverted_image = 1.0 - (
        image / 255.0
    )  # Assuming the image values are in [0, 255] range
    # Convert back to uint8
    inverted_image = (inverted_image * 255).astype(np.uint8)

    # Save the inverted image
    cv2.imwrite(output_file_path, inverted_image)
    print(f"Processed and saved: {output_file_path}")


def get_sep_face_mask(file_path1, file_path2, output_file_path):
    """
    Read two images, subtract the second one from the first, and save the result.

    Parameters:
    output_dir (str): Path to the output directory to save the subtracted image.
    """

    # Read the images
    mask1 = cv2.imread(file_path1, cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(file_path2, cv2.IMREAD_GRAYSCALE)

    if mask1 is None or mask2 is None:
        print(f"Failed to load images: {file_path1}")
        return

    # Ensure the images are the same size
    if mask1.shape != mask2.shape:
        print(
            f"Image shapes do not match for {file_path1}: {mask1.shape} vs {mask2.shape}"
        )
        return

    # Subtract the second mask from the first
    result_mask = cv2.subtract(mask1, mask2)

    # Save the result mask image
    cv2.imwrite(output_file_path, result_mask)
    print(f"Processed and saved: {output_file_path}")

def resample_audio(input_audio_file: str, output_audio_file: str, sample_rate: int):
    p = subprocess.Popen([
        "ffmpeg", "-y", "-v", "error", "-i", input_audio_file, "-ar", str(sample_rate), output_audio_file
    ])
    ret = p.wait()
    assert ret == 0, "Resample audio failed!"
    return output_audio_file

def get_face_region(image_path: str, detector):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to open image: {image_path}. Skipping...")
            return None, None

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = detector.detect(mp_image)

        # Adjust mask creation for the three-channel image
        mask = np.zeros_like(image, dtype=np.uint8)

        for detection in detection_result.detections:
            bbox = detection.bounding_box
            start_point = (int(bbox.origin_x), int(bbox.origin_y))
            end_point = (int(bbox.origin_x + bbox.width),
                         int(bbox.origin_y + bbox.height))
            cv2.rectangle(mask, start_point, end_point,
                          (255, 255, 255), thickness=-1)

        save_path = image_path.replace("images", "face_masks")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, mask)
        # print(f"Processed and saved {save_path}")
        return image_path, mask
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None


def save_checkpoint(model: torch.nn.Module, save_dir: str, prefix: str, ckpt_num: int, total_limit: int = -1) -> None:
    """
    Save the model's state_dict to a checkpoint file.

    If `total_limit` is provided, this function will remove the oldest checkpoints
    until the total number of checkpoints is less than the specified limit.

    Args:
        model (nn.Module): The model whose state_dict is to be saved.
        save_dir (str): The directory where the checkpoint will be saved.
        prefix (str): The prefix for the checkpoint file name.
        ckpt_num (int): The checkpoint number to be saved.
        total_limit (int, optional): The maximum number of checkpoints to keep.
            Defaults to None, in which case no checkpoints will be removed.

    Raises:
        FileNotFoundError: If the save directory does not exist.
        ValueError: If the checkpoint number is negative.
        OSError: If there is an error saving the checkpoint.
    """

    if not osp.exists(save_dir):
        raise FileNotFoundError(
            f"The save directory {save_dir} does not exist.")

    if ckpt_num < 0:
        raise ValueError(f"Checkpoint number {ckpt_num} must be non-negative.")

    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit > 0:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            print(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            print(
                f"Removing checkpoints: {', '.join(removing_checkpoints)}"
            )

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint_path = osp.join(
                    save_dir, removing_checkpoint)
                try:
                    os.remove(removing_checkpoint_path)
                except OSError as e:
                    print(
                        f"Error removing checkpoint {removing_checkpoint_path}: {e}")

    state_dict = model.state_dict()
    try:
        torch.save(state_dict, save_path)
        print(f"Checkpoint saved at {save_path}")
    except OSError as e:
        raise OSError(f"Error saving checkpoint at {save_path}: {e}") from e


def init_output_dir(dir_list: List[str]):
    """
    Initialize the output directories.

    This function creates the directories specified in the `dir_list`. If a directory already exists, it does nothing.

    Args:
        dir_list (List[str]): List of directory paths to create.
    """
    for path in dir_list:
        os.makedirs(path, exist_ok=True)


def load_checkpoint(cfg, save_dir, accelerator):
    """
    Load the most recent checkpoint from the specified directory.

    This function loads the latest checkpoint from the `save_dir` if the `resume_from_checkpoint` parameter is set to "latest".
    If a specific checkpoint is provided in `resume_from_checkpoint`, it loads that checkpoint. If no checkpoint is found,
    it starts training from scratch.

    Args:
        cfg: The configuration object containing training parameters.
        save_dir (str): The directory where checkpoints are saved.
        accelerator: The accelerator object for distributed training.

    Returns:
        int: The global step at which to resume training.
    """
    if cfg.resume_from_checkpoint != "latest":
        resume_dir = cfg.resume_from_checkpoint
    else:
        resume_dir = save_dir
    # Get the most recent checkpoint
    dirs = os.listdir(resume_dir)

    dirs = [d for d in dirs if d.startswith("checkpoint")]
    if len(dirs) > 0:
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])
    else:
        accelerator.print(
            f"Could not find checkpoint under {resume_dir}, start training from scratch")
        global_step = 0

    return global_step


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/
            521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/
    #              521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def extract_audio_from_videos(video_path: Path, audio_output_path: Path) -> Path:
    """
    Extract audio from a video file and save it as a WAV file.

    This function uses ffmpeg to extract the audio stream from a given video file and saves it as a WAV file
    in the specified output directory.

    Args:
        video_path (Path): The path to the input video file.
        output_dir (Path): The directory where the extracted audio file will be saved.

    Returns:
        Path: The path to the extracted audio file.

    Raises:
        subprocess.CalledProcessError: If the ffmpeg command fails to execute.
    """
    ffmpeg_command = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-vn', '-acodec',
        "pcm_s16le", '-ar', '16000', '-ac', '2',
        str(audio_output_path)
    ]

    try:
        print(f"Running command: {' '.join(ffmpeg_command)}")
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio from video: {e}")
        raise

    return audio_output_path


def convert_video_to_images(video_path: Path, output_dir: Path) -> Path:
    """
    Convert a video file into a sequence of images.

    This function uses ffmpeg to convert each frame of the given video file into an image. The images are saved
    in a directory named after the video file stem under the specified output directory.

    Args:
        video_path (Path): The path to the input video file.
        output_dir (Path): The directory where the extracted images will be saved.

    Returns:
        Path: The path to the directory containing the extracted images.

    Raises:
        subprocess.CalledProcessError: If the ffmpeg command fails to execute.
    """
    ffmpeg_command = [
        'ffmpeg',
        '-i', str(video_path),
        '-vf', 'fps=25',
        str(output_dir / '%04d.png')
    ]

    try:
        print(f"Running command: {' '.join(ffmpeg_command)}")
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting video to images: {e}")
        raise

    return output_dir


def get_union_mask(masks):
    """
    Compute the union of a list of masks.

    This function takes a list of masks and computes their union by taking the maximum value at each pixel location.
    Additionally, it finds the bounding box of the non-zero regions in the mask and sets the bounding box area to white.

    Args:
        masks (list of np.ndarray): List of masks to be combined.

    Returns:
        np.ndarray: The union of the input masks.
    """
    union_mask = None
    for mask in masks:
        if union_mask is None:
            union_mask = mask
        else:
            union_mask = np.maximum(union_mask, mask)

    if union_mask is not None:
        # Find the bounding box of the non-zero regions in the mask
        rows = np.any(union_mask, axis=1)
        cols = np.any(union_mask, axis=0)
        try:
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
        except Exception as e:
            print(str(e))
            return 0.0

        # Set bounding box area to white
        union_mask[ymin: ymax + 1, xmin: xmax + 1] = np.max(union_mask)

    return union_mask


def move_final_checkpoint(save_dir, module_dir, prefix):
    """
    Move the final checkpoint file to the save directory.

    This function identifies the latest checkpoint file based on the given prefix and moves it to the specified save directory.

    Args:
        save_dir (str): The directory where the final checkpoint file should be saved.
        module_dir (str): The directory containing the checkpoint files.
        prefix (str): The prefix used to identify checkpoint files.

    Raises:
        ValueError: If no checkpoint files are found with the specified prefix.
    """
    checkpoints = os.listdir(module_dir)
    checkpoints = [d for d in checkpoints if d.startswith(prefix)]
    checkpoints = sorted(
        checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
    )
    shutil.copy2(os.path.join(
        module_dir, checkpoints[-1]), os.path.join(save_dir, prefix + '.pth'))
