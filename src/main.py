import time

import cv2
import pyautogui
import numpy as np

from displayer import create_overlay
from distancer import initialize_battle_params, process_image, print_battle_params

def main() -> None:
    time.sleep(2)
    screen = pyautogui.screenshot()
    screen_array = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)

    battle_params = initialize_battle_params(screen_array)
    print_battle_params(battle_params)

    initial_text = f"M: {battle_params["square_width_meter"]}"
    root, label = create_overlay(initial_text)
    root.after(0, process_image, root, label, battle_params)
    root.mainloop()


if __name__ == "__main__":
    main()
