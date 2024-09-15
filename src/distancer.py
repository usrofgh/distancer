import pyautogui
import pytesseract

import PIL.Image
from math import sqrt
import cv2
import numpy as np

from config import BLACK, METER_LOW, METER_HIGH, ME_LOW, ME_HIGH, POINT_LOW, POINT_HIGH
from displayer import update_tk_text

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def detect_bottom_right_square_width_black_pixel(screen_array: np.array) -> tuple:
    for i_y, y in enumerate(screen_array[::-1]):
        for i_x, p_color in enumerate(y[::-1]):
            if np.all(p_color == BLACK):
                necessary_y = screen_array.shape[0] - (i_y + 1)  # including
                necessary_x = screen_array.shape[1] - (i_x + 1)  # including
                return  necessary_y, necessary_x

def determine_square_pixel_width(screen_array: np.array, start_point: tuple) -> int:
    i_y, i_x = start_point
    while True:
        i_x -= 1
        pix_color = screen_array[i_y][i_x]
        if np.all(pix_color != BLACK):
            square_width = start_point[1] - i_x
            return square_width

def determine_square_meter_width(screen_array: np.array, start_point: tuple, square_width_pixel: int) -> float:
    meter_interest_area = screen_array[
      start_point[0] - square_width_pixel // 3: start_point[0],
      start_point[1] - square_width_pixel: start_point[1]
    ]
    mask = cv2.inRange(meter_interest_area, METER_LOW, METER_HIGH)
    mask_pil = PIL.Image.fromarray(mask)
    square_width_meter = pytesseract.image_to_string(mask_pil, config='--psm 6')
    square_width_meter = int("".join([ch for ch in square_width_meter if ch.isdigit()]))
    if square_width_meter > 1000:
        square_width_meter = int(square_width_meter / 10)
    return square_width_meter

def detect_me_from_drone(map_array: np.array) -> tuple:
    me_mask = cv2.inRange(map_array, ME_LOW, ME_HIGH)
    me_contours, _ = cv2.findContours(me_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in me_contours:
        m = cv2.moments(contour)
        if m["m00"] != 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            me_xy = (cx, cy)
            return me_xy
            # cv2.circle(map_array, (cx, cy), 15, (0, 0, 255), -1)

def detect_point_from_drone(map_array: np.array) -> tuple:
    point_mask = cv2.inRange(map_array, POINT_LOW, POINT_HIGH)
    point_contours, _ = cv2.findContours(point_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in point_contours:
        m = cv2.moments(contour)
        if m["m00"] != 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            point_xy = (cx, cy)
            return point_xy
            # cv2.circle(map_array, (cx, cy), 15, (0, 0, 255), -1)

def initialize_battle_params(screen_array: np.array) -> dict:
    # screen_array = cv2.imread("images/Screenshot_1.png")
    start_point = detect_bottom_right_square_width_black_pixel(screen_array)
    square_width_pixel = determine_square_pixel_width(screen_array, start_point)
    square_width_meter = determine_square_meter_width(screen_array, start_point, square_width_pixel)
    meters_in_pixel = square_width_meter / square_width_pixel

    return {
        "start_point": start_point,
        "square_width_pixel": square_width_pixel,
        "square_width_meter": square_width_meter,
        "meters_in_pixel": meters_in_pixel
    }

def calc_distance(me_xy: tuple, point_xy: tuple, meters_in_pixel: float) -> float:
    try:
        d = sqrt((point_xy[0] - me_xy[0]) ** 2 + (point_xy[1] - me_xy[1]) ** 2)
        d = round(d * meters_in_pixel, 2)
        return d
    except TypeError:
        pass


def print_battle_params(battle_params: dict) -> None:
    square_width_pixel = battle_params.get('square_width_pixel')
    square_width_meter = battle_params.get('square_width_meter')
    meters_in_pixel = battle_params.get('meters_in_pixel')

    print(f"SQUARE WIDTH PIXEL: {square_width_pixel}")
    print(f"SQUARE WIDTH METER: {square_width_meter}")
    print(f"METERS IN PIXEL: {round(meters_in_pixel, 2)}")


def process_image(root, label, battle_params: dict, prev_distance = None):
    screen = pyautogui.screenshot()
    map_array = cv2.cvtColor(np.array(screen.crop((1473, 633, 1907, 1067))), cv2.COLOR_RGB2BGR)

    me_xy = detect_me_from_drone(map_array)
    point_xy = detect_point_from_drone(map_array)
    curr_distance = calc_distance(me_xy, point_xy, battle_params.get('meters_in_pixel'))
    if curr_distance and curr_distance != prev_distance:
        new_text = f"D: {str(curr_distance)}\nM: {battle_params['square_width_meter']}"
        update_tk_text(label, new_text)
        prev_distance = curr_distance

    root.after(200, process_image, root, label, battle_params, prev_distance)
