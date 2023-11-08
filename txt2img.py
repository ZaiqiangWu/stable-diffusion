import torch

from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

pipe = pipe.to("cuda")

prompt = "a photo of an topless girl"
image = pipe(prompt).images[0]

image.save("astronaut_rides_horse.png")