from inference import inference_process
import gradio as gr
def predict(image, audio, filepath, pose_weight, face_weight, lip_weight, face_expand_ratio):
  return inference_process(
    source_image=image, 
    driving_audio=audio,
    output=filepath,
    pose_weight=pose_weight,
    face_weight=face_weight,
    lip_weight=lip_weight,
    face_expand_ratio=face_expand_ratio
  )
app = gr.Interface(
    fn=predict,
    inputs=[
      gr.Image,
      gr.Audio,
      gr.Number(label="pose weight", value=1.0),
      gr.Number(label="face weight", value=1.0),
      gr.Number(label="lip weight", value=1.0)
      gr.Number(label="face expand ratio", value=1.2),
    ],
    outputs=[gr.Video(value=filepath)],
)

app.launch()
