import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe= pipe.to("cuda")

prompt = "Spiderman is surfing"
video_frames = pipe(prompt, num_frames=64).frames[0]
video_path=export_to_video(video_frames, output_video_path="spiderman_surfing.mp4")