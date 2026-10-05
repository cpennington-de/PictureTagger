import tkinter as tk

root = tk.Tk()
root.geometry("400x300")

# Try a Text widget instead
text = tk.Text(root, width=30, height=5, bg="white", fg="black")
text.pack(pady=10)

root.mainloop()
