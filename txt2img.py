from diffusers import DiffusionPipeline
import torch 

base_model_id = "stabilityai/stable-diffusion-xl-base-0.9"
pipeline = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights(".", weight_name="Kamepan.safetensors")

prompt = "anime screencap, glint, drawing, best quality, light smile, shy, a full body of a girl wearing wedding dress in the middle of the forest beneath the trees, fireflies, big eyes, 2d, cute, anime girl, waifu, cel shading, magical girl, vivid colors, (outline:1.1), manga anime artstyle, masterpiece, offical wallpaper, glint <lora:kame_sdxl_v2:1>"
negative_prompt = "(deformed, bad quality, sketch, depth of field, blurry:1.1), grainy, bad anatomy, bad perspective, old, ugly, realistic, cartoon, disney, bad propotions"
generator = torch.manual_seed(2947883060)
num_inference_steps = 30
guidance_scale = 7

image = pipeline(
    prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps,
    generator=generator, guidance_scale=guidance_scale
).images[0]
image.save("Kamepan.png")