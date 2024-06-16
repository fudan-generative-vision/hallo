from inference import inference_process
import argparse
import gradio as gr
from omegaconf import OmegaConf
def predict(image, audio, size, steps, fps, cfg, pose_weight, face_weight, lip_weight, face_expand_ratio):
  dict = {
    'data': {
      'source_image': {
        'width': size,
        'height': size
      },
      'export_video': {
        'fps': fps
      }
    },
    'cfg_scale': cfg,
    'source_image': image,
    'driving_audio': audio,
    'pose_weight': pose_weight,
    'face_weight': face_weight,
    'lip_weight': lip_weight,
    'face_expand_ratio': face_expand_ratio,
    'config': 'configs/inference/default.yaml',
    'checkpoint': None,
    'output': ".cache/output.mp4",
    'inference_steps': steps
  }
  args = argparse.Namespace()
  for key, value in dict.items():
      setattr(args, key, value)
  return inference_process(args)

app = gr.Interface(
    fn=predict,
    inputs=[
      gr.Image(type="filepath"),
      gr.Audio(type="filepath"),
      gr.Number(label="size", value=512, minimum=256, maximum=512, step=64, precision=0),
      gr.Number(label="steps", value=40, minimum=1, step=1, precision=0),
      gr.Number(label="fps", value=25, minimum=1, step=1, precision=0),
      gr.Slider(label="CFG Scale", value=3.5, minimum=0, maximum=10, step=0.01),
      gr.Number(label="pose weight", value=1.0),
      gr.Number(label="face weight", value=1.0),
      gr.Number(label="lip weight", value=1.0),
      gr.Number(label="face expand ratio", value=1.2),
    ],
    outputs=[gr.Video()],
)
app.launch()
