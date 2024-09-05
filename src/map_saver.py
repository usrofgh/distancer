from mss import mss
from pathlib import Path
import os

# Directory where the image will be saved
save_directory = Path().resolve() / "images"
os.makedirs(save_directory, exist_ok=True)

def capture_area():
    left, top, right, bottom = 1426, 706, 1905, 1185
    width = right - left
    height = bottom - top

    with mss() as sct:
        monitor = {"top": top, "left": left, "width": width, "height": height}
        screenshot = sct.grab(monitor)

        # Save the image to the specified directory
        file_path = os.path.join(save_directory, "rename.png")
        mss.tools.to_png(screenshot.rgb, screenshot.size, output=file_path)

capture_area()
