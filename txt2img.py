# モデルのアーキテクチャを定義する
import torch
from stable_diffusion import StableDiffusionPipeline, DPMSolverMultistepScheduler
model_id = "stabilityai/stable-diffusion-2-1"
model = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
model.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# .ckptファイルからモデルのパラメータを読み込む
ckpt_path = './sd-v1-5-inpainting.ckpt'
checkpoint = torch.load(ckpt_path, map_location="gpu")
model.load_state_dict(checkpoint['state_dict'])

# モデルを推論モードにする
model.eval()

# モデルにプロンプトを与えて、画像を生成する
prompt = 'xxxxxx'
image = model(prompt)
