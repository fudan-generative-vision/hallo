from inference import inference_process
import argparse
import gradio as gr
def predict(image, audio, pose_weight, face_weight, lip_weight, face_expand_ratio):
  args = argparse.Namespace()
  args.source_image=image
  args.driving_audio=audio
  args.pose_weight=pose_weight
  args.face_weight=face_weight
  args.lip_weight=lip_weight
  args.face_expand_ratio=face_expand_ratio
  args.config = "configs/inference/default.yaml"
  return inference_process(args)
app = gr.Interface(
    fn=predict,
    inputs=[
      gr.Image(type="filepath"),
      gr.Audio(type="filepath"),
      gr.Number(label="pose weight", value=1.0),
      gr.Number(label="face weight", value=1.0),
      gr.Number(label="lip weight", value=1.0),
      gr.Number(label="face expand ratio", value=1.2),
    ],
    outputs=[gr.Video()],
)

app.launch()
