import time

import pyautogui
import pytesseract

import PIL.Image
from math import sqrt
import cv2
import numpy as np

from displayer import create_overlay, update_tk_text

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

SQUARE_WIDTH_PIXEL = None
SQUARE_WIDTH_METER = None
METERS_IN_PIXEL = None
START_POINT = None
CROP = None
PREV_DISTANCE = None

BLACK_BGR = np.array([0, 0, 0], dtype=np.uint8)

METER_LOW = np.array([0, 0, 0], dtype=np.uint8)
METER_HIGH = np.array([37, 38, 38], dtype=np.uint8)

ME_LOW = np.array([60, 151, 72], dtype=np.uint8)
ME_HIGH = np.array([86, 215, 103], dtype=np.uint8)
POINT_LOW = np.array([5, 153, 155], dtype=np.uint8)
POINT_HIGH = np.array([7, 216, 216], dtype=np.uint8)


def initialize_params(screen_array: np.array):
    global START_POINT
    if START_POINT:
        return

    global SQUARE_WIDTH_PIXEL
    global SQUARE_WIDTH_METER
    global METERS_IN_PIXEL

    # ----- DETECT BOTTOM RIGHT BLACK_BGR PIXEL COORDINATE OF SQUARE WIDTH -----
    is_exit = False
    for i_row, row in enumerate(screen_array[::-1]):
        for i_column, p_color in enumerate(row[::-1]):
            if np.all(p_color == BLACK_BGR):
                source_row = screen_array.shape[0] - (i_row + 1)
                source_column = screen_array.shape[1] - (i_column + 1)
                START_POINT = (source_column, source_row)
                is_exit = True
                break
        if is_exit:
            break
    if not START_POINT:
        return

    # ----------------------------------------------------------------------

    # ----- DETECT SQUARE WIDTH IN PIXELS -----
    i_square_pixel_width = START_POINT[0]
    j_square_pixel_width = START_POINT[1]
    while ...:
        i_square_pixel_width -= 1
        if np.all(screen_array[j_square_pixel_width][i_square_pixel_width] != BLACK_BGR):
            SQUARE_WIDTH_PIXEL = START_POINT[0] - i_square_pixel_width
            print(f"SQUARE WITH PIXEL: {SQUARE_WIDTH_PIXEL}")
            break
    # ----------------------------------------

    # ----- DETECT SQUARE WIDTH IN METERS -----
    meter_interest_area = screen_array[
                          START_POINT[1] - SQUARE_WIDTH_PIXEL // 3: START_POINT[1],
                          START_POINT[0] - SQUARE_WIDTH_PIXEL: START_POINT[0]
                          ]
    mask = cv2.inRange(meter_interest_area, METER_LOW, METER_HIGH)
    mask_pil = PIL.Image.fromarray(mask)
    SQUARE_WIDTH_METER = pytesseract.image_to_string(mask_pil, config='--psm 6')
    try:
        SQUARE_WIDTH_METER = int("".join([ch for ch in SQUARE_WIDTH_METER if ch.isdigit()]))
        print(f"SQUARE WITH METER: {SQUARE_WIDTH_METER}")
    except ValueError:
        START_POINT = None
        SQUARE_WIDTH_PIXEL = None
        SQUARE_WIDTH_METER = None
        METERS_IN_PIXEL = None
    # --------------------------------------

    METERS_IN_PIXEL = SQUARE_WIDTH_METER / SQUARE_WIDTH_PIXEL
    print(f"METERS IN PIXEL: {round(METERS_IN_PIXEL, 2)}")

def calc(map_array: np.array):
    # ----- DETECT ME CENTER FROM A DRONE -----
    me_mask = cv2.inRange(map_array, ME_LOW, ME_HIGH)
    me_contours, _ = cv2.findContours(me_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    me_xy = None
    for contour in me_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            me_xy = (cx, cy)
            cv2.circle(map_array, (cx, cy), 15, (0, 0, 255), -1)
    # -----------------------------------------

    # ----- DETECT YELLOW POINT CENTER FROM A DRONE -----
    point_mask = cv2.inRange(map_array, POINT_LOW, POINT_HIGH)
    point_contours, _ = cv2.findContours(point_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    point_xy = None
    for contour in point_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            point_xy = (cx, cy)
            cv2.circle(map_array, (cx, cy), 15, (0, 0, 255), -1)
    # -----------------------------------------

    # print(f"ME: {me_xy} |    POINT: {point_xy}")
    try:
        d = sqrt((point_xy[0] - me_xy[0]) ** 2 + (point_xy[1] - me_xy[1]) ** 2)
        d = round(d * METERS_IN_PIXEL, 2)
        return d
    except TypeError:
        pass


def process_image(root, label):
    global PREV_DISTANCE
    screen = pyautogui.screenshot()
    map_array = cv2.cvtColor(np.array(screen.crop((1473, 633, 1907, 1067))), cv2.COLOR_RGB2BGR)
    curr_distance = calc(map_array)
    if curr_distance and curr_distance != PREV_DISTANCE:
        new_text = f"D: {str(curr_distance)}\nM: {SQUARE_WIDTH_METER}"
        update_tk_text(label, new_text)
        # print(f"\r{curr_distance}", end="")
        PREV_DISTANCE = curr_distance
    root.after(200, process_image, root, label)

def main():
    time.sleep(2)
    screen = pyautogui.screenshot()
    screen_array = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
    initialize_params(screen_array)

    root, label = create_overlay()
    root.after(0, process_image, root, label)
    root.mainloop()


if __name__ == "__main__":
    main()
