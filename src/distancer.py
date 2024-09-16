import time
from copy import deepcopy
import pyautogui
import keyboard

import pytesseract

import PIL.Image
from math import sqrt
import cv2
import numpy as np

from config import BLACK, METER_LOW, METER_HIGH, ME_LOW, ME_HIGH, POINT_LOW, POINT_HIGH, MIN_BORDER_COLOR_MINIMAP, \
MAX_BORDER_COLOR_MINIMAP, IS_BIG_MAP
from displayer import update_tk_text

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def detect_bottom_right_square_width_black_pixel(screen_array: np.array) -> tuple:
    for i_y, y in enumerate(screen_array[::-1]):
        for i_x, p_color in enumerate(y[::-1]):
            if np.all(p_color == BLACK):
                necessary_y = screen_array.shape[0] - (i_y + 1)  # including
                necessary_x = screen_array.shape[1] - (i_x + 1)  # including
                return  necessary_y, necessary_x


def determine_minimap_borders(screen_array: np.array, start_point: tuple) -> tuple:
    bottom_right_map = None
    bottom_left_map = None
    top_right_map = None
    top_left_map = None
    curr_point = list(start_point)
    curr_point[0] += 1
    while ...:
        y, x = curr_point
        if np.all(np.logical_and(screen_array[y][x] >= MIN_BORDER_COLOR_MINIMAP, screen_array[y][x] <= MAX_BORDER_COLOR_MINIMAP)):
            curr_point[1] += 1
        else:
            curr_point[1] -= 1
            x -= 1
            bottom_right_map = [y, x]
            break

    while ...:
        y, x = curr_point
        if np.all(np.logical_and(screen_array[y][x] >= MIN_BORDER_COLOR_MINIMAP, screen_array[y][x] <= MAX_BORDER_COLOR_MINIMAP)):
            curr_point[0] -= 1
        else:
            curr_point[0] += 1
            y += 1
            top_right_map = [y, x]
            break

    while ...:
        y, x = curr_point
        if np.all(np.logical_and(screen_array[y][x] >= MIN_BORDER_COLOR_MINIMAP, screen_array[y][x] <= MAX_BORDER_COLOR_MINIMAP)):
            curr_point[1] -= 1
        else:
            curr_point[1] += 1
            x += 1
            top_left_map = [y, x]
            break

    while ...:
        y, x = curr_point
        if np.all(np.logical_and(screen_array[y][x] >= MIN_BORDER_COLOR_MINIMAP, screen_array[y][x] <= MAX_BORDER_COLOR_MINIMAP)):
            curr_point[0] += 1
        else:
            curr_point[0] -= 1
            y -= 1
            bottom_left_map = [y, x]
            break

    return top_left_map, bottom_right_map


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
    start_point = detect_bottom_right_square_width_black_pixel(screen_array)
    square_width_pixel = determine_square_pixel_width(screen_array, start_point)
    square_width_meter = determine_square_meter_width(screen_array, start_point, square_width_pixel)
    meters_in_pixel = square_width_meter / square_width_pixel

    # minimap_borders = determine_minimap_borders(screen_array, start_point)  # TODO: gradients
    minimap_borders = ([633, 1473], [1066, 1906])

    return {
        "start_point": start_point,
        "square_width_pixel": square_width_pixel,
        "square_width_meter": square_width_meter,
        "meters_in_pixel": meters_in_pixel,
        "minimap_borders": minimap_borders,
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
    minimap_borders = battle_params.get('minimap_borders')

    print(f"SQUARE WIDTH PIXEL: {square_width_pixel}")
    print(f"SQUARE WIDTH METER: {square_width_meter}")
    print(f"METERS IN PIXEL: {round(meters_in_pixel, 2)}")
    print(f"MINIMAP BORDERS: {minimap_borders}")


def process_image(
    root,
    label,
    battle_params: dict,
    prev_minimap_distance = None,
    prev_minimap_me_xy = None,
    prev_minimap_point_xy = None
):
    screen = pyautogui.screenshot()
    screen = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)

    minimap_borders = battle_params["minimap_borders"]
    minimap_array = screen[
        minimap_borders[0][0]:minimap_borders[1][0] + 1,
        minimap_borders[0][1]:minimap_borders[1][1] + 1,
    ]

    minimap_point_xy = detect_point_from_drone(minimap_array)
    minimap_me_xy = detect_me_from_drone(minimap_array)

    if (
        (minimap_point_xy is None or minimap_me_xy is None) or
        (prev_minimap_me_xy == minimap_me_xy and prev_minimap_point_xy == minimap_point_xy)
    ):
        root.after(200, process_image, root, label, battle_params, prev_minimap_distance, prev_minimap_me_xy, prev_minimap_point_xy)
        # root.after(200, process_image, root, label, battle_params, prev_minimap_distance)
        return

    full_map_curr_distance = None
    if IS_BIG_MAP:
        x = 782
        y = 178
        width = x + 844 + 1
        height = y + 844 + 1
        crop = (x, y, width, height)
        keyboard.press("m")
        time.sleep(0.3)
        full_map_screen = pyautogui.screenshot().crop(crop)
        keyboard.release("m")

        full_map_array = cv2.cvtColor(np.array(full_map_screen), cv2.COLOR_RGB2BGR)
        full_map_point_xy = detect_point_from_drone(full_map_array)
        full_map_me_xy = detect_me_from_drone(full_map_array)
        full_map_curr_distance = calc_distance(full_map_me_xy, full_map_point_xy, battle_params.get('meters_in_pixel'))


    minimap_curr_distance = calc_distance(minimap_me_xy, minimap_point_xy, battle_params.get('meters_in_pixel'))

    if minimap_curr_distance and minimap_curr_distance != prev_minimap_distance:
        d = full_map_curr_distance if IS_BIG_MAP else minimap_curr_distance
        # d = minimap_curr_distance
        new_text = f"D: {str(d)}\nM: {battle_params['square_width_meter']}"
        update_tk_text(label, new_text)
        root.after(200, process_image, root, label, battle_params, minimap_curr_distance, minimap_me_xy, minimap_point_xy)
        # root.after(200, process_image, root, label, battle_params, minimap_curr_distance)
    else:
        root.after(200, process_image, root, label, battle_params, prev_minimap_distance, prev_minimap_me_xy, prev_minimap_point_xy)
        # root.after(200, process_image, root, label, battle_params, prev_minimap_distance)
