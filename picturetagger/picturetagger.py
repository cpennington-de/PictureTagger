import torch
from transformers import AutoModelForCausalLM
from PIL import Image

# Make sure MPS is available
if not torch.backends.mps.is_available():
    raise SystemError("MPS (Metal Performance Shaders) is not available on this machine.")

device = torch.device("mps")

# Load model on MPS
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True
).to(device)

# Load and process image
image = Image.open("/Users/colinpennington/Pictures/IMG_1412.JPG")

# Captioning
print("Short caption:")
print(model.caption(image, length="short")["caption"])

print("Detailed caption:")
for t in model.caption(image, length="normal", stream=True)["caption"]:
    print(t, end="", flush=True)

print("Tags:")
print(model.query(image, "Generate 5 tags for the image")["answer"])

print("Title:")
print(model.query(image,"Create a Title fit for a stock photo listing")["answer"])