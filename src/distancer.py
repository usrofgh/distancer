import time
from datetime import datetime

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


def filter_contours_by_distance(contours, max_distance=20):
    def distance_between_contours(c1, c2):
        c1_points = c1.reshape(-1, 2)
        c2_points = c2.reshape(-1, 2)
        distances = [np.linalg.norm(p1 - p2) for p1 in c1_points for p2 in c2_points]
        return np.min(distances)
    filtered_contours = []
    for i, c1 in enumerate(contours):
        for j, c2 in enumerate(contours):
            if i != j and distance_between_contours(c1, c2) <= max_distance:
                if i not in filtered_contours:
                    filtered_contours.append(i)
                if j not in filtered_contours:
                    filtered_contours.append(j)

    return [contours[i] for i in filtered_contours]
def get_img_without_internal_contour(img: np.array) -> np.array:
    point_contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    point_contours = filter_contours_by_distance(point_contours)
    if not point_contours or (len(point_contours) < 2 and cv2.contourArea(max(point_contours, key=cv2.contourArea)) < 10):
        return None

    external_image = np.zeros_like(img)

    for i, contour in enumerate(point_contours):
        x, y, w, h = cv2.boundingRect(contour)
        is_inner = False
        for j, other_contour in enumerate(point_contours):
            if i != j:
                x2, y2, w2, h2 = cv2.boundingRect(other_contour)
                if x > x2 and y > y2 and (x + w) < (x2 + w2) and (y + h) < (y2 + h2):
                    is_inner = True
                    break
        if not is_inner:
            cv2.drawContours(external_image, [contour], -1, 255, thickness=1)

    kernel = np.ones((3, 3), np.uint8)
    closed_image = cv2.morphologyEx(external_image, cv2.MORPH_CLOSE, kernel)
    return closed_image


def detect_point_from_drone(map_array: np.array) -> tuple | None:
    point_mask = cv2.inRange(map_array, POINT_LOW, POINT_HIGH)
    ext_mask = get_img_without_internal_contour(point_mask)
    if ext_mask is None:
        return None, None, None

    ext_contours, _ = cv2.findContours(ext_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest_contour = max(ext_contours, key=cv2.contourArea)
    hull = cv2.convexHull(biggest_contour)
    (x, y), radius = cv2.minEnclosingCircle(hull)
    if radius < 8:
        return None, None, None
    # else:
    #     return round(x), round(y)


    circle_mask = np.zeros_like(ext_mask)
    cv2.circle(circle_mask, (int(x), int(y)), int(radius), 255, 1)
    point_contours, _ = cv2.findContours(circle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    biggest_contour = max(point_contours, key=cv2.contourArea)
    m = cv2.moments(biggest_contour)
    if m["m00"] != 0:
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        point_xy = (cx, cy)
        contour_pixels = tuple(point[0] for point in biggest_contour)
        return point_xy, contour_pixels, circle_mask
    else:
        return None, None, None

def is_shifted(previous_contour_pixels: tuple, current_contour_pixels: tuple) -> bool:
    previous_set = set(map(tuple, [tuple(point) for point in previous_contour_pixels]))
    current_set = set(map(tuple, [tuple(point) for point in current_contour_pixels]))

    matching_pixels = previous_set.intersection(current_set)
    num_matching_pixels = len(matching_pixels)
    num_previous_pixels = len(previous_set)


    if num_matching_pixels > (num_previous_pixels * 0.5):
        shift_detected = False
    else:
        shift_detected = True

    return shift_detected

def get_battle_id(screen_array: np.array) -> str:
    ...

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

def is_points_close(point1, point2, tolerance=1):
    return abs(point1[0] - point2[0]) <= tolerance and abs(point1[1] - point2[1]) <= tolerance

def process_image(
    root,
    label,
    battle_params: dict,
    prev_minimap_distance = None,
    prev_minimap_me_xy = None,
    prev_minimap_point_xy = None,
    prev_point_xy = None
):
    try:
        screen = pyautogui.screenshot()
    except Exception:
        root.after(100, process_image, root, label, battle_params, prev_minimap_distance, prev_minimap_me_xy, prev_minimap_point_xy, prev_point_xy)
        return

    screen = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)

    minimap_borders = battle_params["minimap_borders"]
    minimap_array = screen[
        minimap_borders[0][0]:minimap_borders[1][0] + 1,
        minimap_borders[0][1]:minimap_borders[1][1] + 1,
    ]

    minimap_point_xy, contours, cyrcle_mask = detect_point_from_drone(minimap_array)
    minimap_me_xy = detect_me_from_drone(minimap_array)

    if (
        (minimap_point_xy is None or minimap_me_xy is None) or
        (prev_minimap_me_xy == minimap_me_xy and is_points_close(prev_minimap_point_xy, minimap_point_xy))
    ):
        root.after(100, process_image, root, label, battle_params, prev_minimap_distance, prev_minimap_me_xy, prev_minimap_point_xy, prev_point_xy)
        return

    img_name = f"{str(datetime.now().timestamp()).replace('.', '')}"
    point_mask = cv2.inRange(minimap_array, POINT_LOW, POINT_HIGH)
    cv2.imwrite(f"point/{img_name}.png", point_mask)
    cv2.imwrite(f"point/{img_name}_morph.png", cyrcle_mask)
    cv2.imwrite(f"point/{img_name}_1.png", minimap_array)

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
        full_map_point_xy, contours, cyrcle_mask = detect_point_from_drone(full_map_array)
        full_map_me_xy = detect_me_from_drone(full_map_array)
        full_map_curr_distance = calc_distance(full_map_me_xy, full_map_point_xy, battle_params.get('meters_in_pixel'))
        if not full_map_curr_distance:
            root.after(100, process_image, root, label, battle_params, prev_minimap_distance, prev_minimap_me_xy,
                prev_minimap_point_xy, prev_point_xy
            )
            return

    minimap_curr_distance = calc_distance(minimap_me_xy, minimap_point_xy, battle_params.get('meters_in_pixel'))

    if minimap_curr_distance and minimap_curr_distance != prev_minimap_distance:
        d = full_map_curr_distance if IS_BIG_MAP else minimap_curr_distance
        new_text = f"D: {str(d)}\nM: {battle_params['square_width_meter']}"
        update_tk_text(label, new_text)
        root.after(100, process_image, root, label, battle_params, minimap_curr_distance, minimap_me_xy, minimap_point_xy)
    else:
        root.after(100, process_image, root, label, battle_params, prev_minimap_distance, prev_minimap_me_xy, prev_minimap_point_xy, prev_point_xy)
