import os
import csv
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

file_paths = filedialog.askopenfilenames(
    title="Select Images",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *bmp")]

)

print(file_paths)