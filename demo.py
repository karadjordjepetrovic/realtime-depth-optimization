from transformers import pipeline
from PIL import Image

pipe = pipeline("depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf", device=0)  # device=0 = your GPU
img = Image.open("frames/frame_001.png")
out = pipe(img)
out["depth"].save("frame_001_depth.png")
print("done")