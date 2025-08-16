import os
import csv
import tkinter as tk
from tkinter import filedialog
import torch
from transformers import AutoModelForCausalLM
from PIL import Image

# Variables to be put into a row of a CSV
filename = ''
description = ''
keywords = ''
categories = ''
firstline = ['Filename', 'Description', 'Keywords', 'Categories', 'Editorial', 'Mature content', 'Illustration']
nextrow = []

categories_types = """Abstract, Animals/Wildlife, Arts, Backgrounds/Textures, Beauty/Fashion, Buildings/Landmarks,
Business/Finance, Celebrities, Education, Food and drink, Healthcare/Medical, Holidays, Industrial, Interiors, Miscellaneous,
Nature, Objects, Parks/Outdoor, People, Religion, Science, Signs/Symbols, Technology, Transportation, Vintage
"""
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

#print(file_paths)
#print(type(file_paths))

with open('data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(firstline)

for item in file_paths:


    # Load and process image
    image = Image.open(item)
    print("Filename:")
    print(os.path.basename(item))
    filename = os.path.basename(item)

    #print('Description')
    #print(model.query(image, "Generate a 10 work title of the image fit for a stock photo title")["answer"])
    description = model.query(image, "Generate a description of the image fit for a stock photo title, at least 10 words")["answer"]
    

    #print("Keywords:")
    #print(model.query(image, "Generate 8 tags for the image")["answer"])
    keywords = model.query(image, "return a string including 8 tags seperated by commas")["answer"]

    #need to work on the Categories section, Model not producing output needed
    #print('Categories') 
    #print(model.query(image, "pick exactly two from the list (seperated by a , and nothing else)" + categories_types)["answer"])
    #categories = model.query(image, "return a substring of only one category seperated by commas ->" + categories_types)["answer"]
    
    nextrow = [filename, description, keywords, 'nature', 'n', 'n', 'n']
    with open('data.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(nextrow)



