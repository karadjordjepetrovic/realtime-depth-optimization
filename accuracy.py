import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image

name = "depth-anything/Depth-Anything-V2-Small-hf"
proc = AutoImageProcessor.from_pretrained(name)
inp = proc(images=Image.open("frames/frame_001.png"), return_tensors="pt")["pixel_values"].cuda()

m32 = AutoModelForDepthEstimation.from_pretrained(name).cuda().eval()
with torch.inference_mode():
    ref = m32(pixel_values=inp).predicted_depth
    for dt in [torch.float16, torch.bfloat16]:
        m = AutoModelForDepthEstimation.from_pretrained(name).to("cuda", dtype=dt).eval()
        out = m(pixel_values=inp.to(dt)).predicted_depth.float()
        rel = ((out - ref).abs().mean() / ref.abs().mean()).item()
        print(f"{dt}: mean relative error vs FP32 = {rel*100:.3f}%")