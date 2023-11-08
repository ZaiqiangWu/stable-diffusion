from diffusers import DiffusionPipeline
import torch

model_id = "./"
pipe = DiffusionPipeline.from_pretrained(model_id, use_safetensors=True)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]

image.save("astronaut_rides_horse.png")