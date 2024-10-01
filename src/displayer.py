import tkinter as tk


def create_overlay(square_width_meter: int):
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.configure(bg='black')
    root.overrideredirect(True)
    label = tk.Label(root, text=str(square_width_meter), fg='#F0F0F0', bg='black', font=('Arial', 15))
    label.pack()
    root.geometry(f"{100}x{50}+{1473-100}+{633}")
    return root, label

def update_tk_text(label: tk.Label, new_text: str) -> None:
    label.config(text=new_text)
