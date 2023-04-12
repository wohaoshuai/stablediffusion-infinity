from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DEISMultistepScheduler, DiffusionPipeline
import cv2
from PIL import Image
import numpy as np
import torch

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting", custom_pipeline="stable_diffusion_controlnet_inpaint")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    safety_checker=None,
    # revision='fp16',
    # torch_dtype=torch.float16,
    controlnet=ControlNetModel.from_pretrained("thepowefuldeez/sd21-controlnet-canny")
).to('mps')
pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)

image = np.array(Image.open("10.png"))

low_threshold = 100
high_threshold = 200
image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

im = pipe(
    "beautiful woman", image=canny_image, num_inference_steps=30, 
    negative_prompt="ugly, blurry, bad, deformed, bad anatomy", 
    generator=torch.manual_seed(42)
).images[0]
