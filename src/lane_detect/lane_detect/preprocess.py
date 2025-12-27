import cv2
import numpy as np
import time
import configparser

# ================= CONFIG =================

def load_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)

    cfg = {
        "FRAME_W": config.getint("LANE", "FRAME_W"),
        "FRAME_H": config.getint("LANE", "FRAME_H"),
        "SCAN_RATIO": config.getfloat("LANE", "SCAN_RATIO"),
        "CENTER_THRESHOLD": config.getint("LANE", "CENTER_THRESHOLD"),
        "NORMAL_LANE_WIDTH": config.getint("LANE", "NORMAL_LANE_WIDTH"),
        "JUNCTION_FACTOR": config.getfloat("LANE", "JUNCTION_FACTOR"),
        "JUNCTION_COOLDOWN": config.getfloat("LANE", "JUNCTION_COOLDOWN"),
    }

    cfg["SCAN_ROW"] = int(cfg["FRAME_H"] * cfg["SCAN_RATIO"])
    return cfg

# ================= PREPROCESS =================

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur, 100, 255, cv2.THRESH_BINARY_INV
    )
    return thresh

# ================= LANE =================

def extract_lane_edges(thresh, scan_row, last_state):
    row_pixels = thresh[scan_row, :]
    white_pixels = np.where(row_pixels == 255)[0]

    if len(white_pixels) > 0:
        left = white_pixels[0]
        right = white_pixels[-1]
        center = (left + right) // 2
        detected = True
    else:
        detected = False
        left, right, center = last_state

    return detected, left, right, center

# ================= JUNCTION =================

def detect_junction(left, right, cfg):
    width = right - left
    return width > cfg["NORMAL_LANE_WIDTH"] * cfg["JUNCTION_FACTOR"]

def junction_hold_logic(raw_state, timer_state, cfg):
    active, start = timer_state
    now = time.time()

    if active:
        if now - start < cfg["JUNCTION_COOLDOWN"]:
            return 1, timer_state
        else:
            return 1, (False, 0.0)

    if raw_state == 1:
        return 1, (True, now)

    return 0, timer_state


