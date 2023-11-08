import torch

from diffusers import DiffusionPipeline, StableDiffusionPipeline


#pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = StableDiffusionPipeline.from_single_file('senhentai_v12.safetensors',local_files_only=True, torch_dtype=torch.float16,safety_checker=None,requires_safety_checker=False)

# "deliberate.safetensors",local_files_only=True, torch_dtype=torch.float16,safety_checker=None,requires_safety_checker=False


def null_safety(images, **kwargs):
    return images, [False]


pipe.safety_checker = null_safety

pipe = pipe.to("cuda")

prompt = "a photo of an topless girl"
image = pipe(prompt).images[0]

image.save("astronaut_rides_horse.png")