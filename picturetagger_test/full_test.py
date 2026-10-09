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
    description = model.query(image, """Generate a concise stock photo description for: {}, {}. 
    Requirements:
    - Exactly 12-18 words
    - Include specific location
    - Start with main subject/action
    - Add relevant details (time of day, weather, mood, composition)
    - Use SEO-friendly keywords
    - Professional tone suitable for stock photo sites""".format(LocationCity, LocationState))["answer"]


    #print("Keywords:")
    #print(model.query(image, "Generate 8 tags for the image")["answer"])
    keywords = model.query(image, """Generate relevant stock photo keywords/tags for this image located in {}, {}.
    Requirements:
    - Separate by commas only (no quotes, no numbering)
    - Include: subject, location, mood/style, composition type
    - Use single words or 2-word phrases
    - Make them searchable and specific
    - Professional/commercial tone
    - Must include the state in at least one keyword (convert 2 character state abbreviation to full state name)
    - Must include the city in at least one keyword
    - Minimum 12 keywords, maximum 20 keywords
    Example format: subject, action, location, mood, style, detail, composition, category""".format(LocationCity, LocationState))["answer"]

    #need to work on the Categories section, Model not producing output needed
    #print('Categories') 
    #print(model.query(image, "pick exactly two from the list (seperated by a , and nothing else)" + categories_types)["answer"])
    categories = model.query(image, """"return exactly one substring of either 'nature', 'urban', 'industrial' 
    "Requirements:
    - Return only one of the three categories: 'nature', 'urban', or 'industrial
    - Do not include any other text, punctuation, or explanation
    - Use lowercase letters only
    - Provide a single word output with no quotes or formatting
    - choose the category that best represents the primary subject and setting of the image
    """)["answer"]
    
    nextrow = [filename, description, keywords, categories, 'n', 'n', 'n']
    with open('data.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(nextrow)


