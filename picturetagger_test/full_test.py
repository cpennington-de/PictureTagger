import os
import csv
import tkinter as tk
from tkinter import filedialog, simpledialog
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
LocationCity = input("Enter Location City: ")
LocationState = input("Enter Location State: ")
file_paths = []

categories_types = """Abstract, Animals/Wildlife, Arts, Backgrounds/Textures, Beauty/Fashion, Buildings/Landmarks,
Business/Finance, Celebrities, Education, Food and drink, Healthcare/Medical, Holidays, Industrial, Interiors, Miscellaneous,
Nature, Objects, Parks/Outdoor, People, Religion, Science, Signs/Symbols, Technology, Transportation, Vintage
"""

def select_files():
    global file_paths
    file_paths = filedialog.askopenfilenames(
        parent=root,
        title="Select Images",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *bmp")]
    )





root = tk.Tk()
root.title("Picture Tagger")
#root.withdraw()
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

tk.Label(root, text="Select Images to Tag")
tk.Button(root, text="Select Images", command=select_files).pack()
tk.Button(root, text="Process Images", command=root.quit).pack()

root.mainloop()




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
    #print(model.query(image, "Generate a 10 to 20 word title of the image fit for a stock photo title")["answer"])
    description = model.query(image, "Generate a description of the image to provide a precise but descriptive title for a stock photo website, at least 10 words max 20 words, must include location, located in " + LocationCity + ", " + LocationState + "}")["answer"]
    

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



