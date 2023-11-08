import torch

from diffusers import DiffusionPipeline


#pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = DiffusionPipeline.from_pretrained("./ckpts/senhentai_v12.safetensors")


def null_safety(images, **kwargs):
    return images, [False]


pipe.safety_checker = null_safety

pipe = pipe.to("cuda")

prompt = "a photo of an topless girl"
image = pipe(prompt).images[0]

image.save("astronaut_rides_horse.png")