import os
import csv
import tkinter as tk
from tkinter import filedialog
import torch
from transformers import AutoModelForCausalLM
from PIL import Image

root = tk.Tk()
root.withdraw()
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

file_paths = filedialog.askopenfilenames(
    title="Select Images",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *bmp")]

)

print(file_paths)
print(type(file_paths))

for item in file_paths:


    # Load and process image
    image = Image.open(item)
    print("Tags:")
    print(model.query(image, "Generate 5 tags for the image")["answer"])