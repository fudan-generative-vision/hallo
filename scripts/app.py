"""
This script is a gradio web ui.

The script takes an image and an audio clip, and lets you configure all the
variables such as cfg_scale, pose_weight, face_weight, lip_weight, etc.

Usage:
This script can be run from the command line with the following command:

python scripts/app.py
"""
import argparse

import gradio as gr
from inference import inference_process


def predict(image, audio, pose_weight, face_weight, lip_weight, face_expand_ratio, progress=gr.Progress(track_tqdm=True)):
    """
    Create a gradio interface with the configs.
    """
    _ = progress
    config = {
        'source_image': image,
        'driving_audio': audio,
        'pose_weight': pose_weight,
        'face_weight': face_weight,
        'lip_weight': lip_weight,
        'face_expand_ratio': face_expand_ratio,
        'config': 'configs/inference/default.yaml',
        'checkpoint': None,
        'output': ".cache/output.mp4"
    }
    args = argparse.Namespace()
    for key, value in config.items():
        setattr(args, key, value)
    return inference_process(args)

app = gr.Interface(
    fn=predict,
    inputs=[
      gr.Image(label="source image (no webp)", type="filepath", format="jpeg"),
      gr.Audio(label="source audio", type="filepath"),
      gr.Number(label="pose weight", value=1.0),
      gr.Number(label="face weight", value=1.0),
      gr.Number(label="lip weight", value=1.0),
      gr.Number(label="face expand ratio", value=1.2),
    ],
    outputs=[gr.Video()],
)
app.launch()
