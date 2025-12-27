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
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
    return thresh


# ================= LANE EXTRACTION =================

def extract_lane_edges(thresh, scan_row, last_state):
    row_pixels = thresh[scan_row, :]
    white_pixels = np.where(row_pixels == 255)[0]

    if len(white_pixels) > 0:
        left_edge = white_pixels[0]
        right_edge = white_pixels[-1]
        lane_center = (left_edge + right_edge) // 2
        detected = True
    else:
        detected = False
        left_edge, right_edge, lane_center = last_state

    return detected, left_edge, right_edge, lane_center


# ================= VISUALIZATION =================

def draw_lane_visual(frame, scan_row, left, right, center, frame_w):
    cv2.line(frame, (0, scan_row), (frame_w, scan_row), (255, 255, 0), 1)
    cv2.circle(frame, (left, scan_row), 5, (0, 255, 0), -1)
    cv2.circle(frame, (right, scan_row), 5, (0, 255, 0), -1)
    cv2.circle(frame, (center, scan_row), 8, (255, 0, 0), -1)


# ================= JUNCTION =================

def detect_junction(left, right, normal_width, factor):
    lane_width = right - left
    return lane_width > normal_width * factor, lane_width


def junction_hold_logic(raw_state, timer_state, cooldown):
    junction_active, start_time = timer_state
    now = time.time()

    if junction_active:
        if now - start_time < cooldown:
            return 1, (junction_active, start_time)
        else:
            return 0, (False, 0.0)

    if raw_state == 1:
        return 1, (True, now)

    return 0, (False, 0.0)


# ================= MAIN DETECTOR =================

def detect_lane(frame, cfg, memory):
    thresh = preprocess_frame(frame)

    detected, left, right, center = extract_lane_edges(
        thresh,
        cfg["SCAN_ROW"],
        memory
    )

    if not detected and memory[0] is None:
        return frame, "no_lane", 0, thresh, memory

    draw_lane_visual(frame, cfg["SCAN_ROW"], left, right, center, cfg["FRAME_W"])

    is_junction, lane_width = detect_junction(
        left,
        right,
        cfg["NORMAL_LANE_WIDTH"],
        cfg["JUNCTION_FACTOR"]
    )

    lane_info = "junction" if is_junction else "straight"
    raw_state = 1 if is_junction else 0

    if is_junction:
        cv2.putText(
            frame,
            "JUNCTION DETECTED",
            (left, cfg["SCAN_ROW"] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    new_memory = (left, right, center)
    return frame, lane_info, raw_state, thresh, new_memory


# ================= MAIN LOOP =================

def main():
    config_path = r"C:\Users\ADMIN\Desktop\osaka_MLS\line\src\config.ini"
    video_path = r"C:\Users\ADMIN\Desktop\osaka_MLS\line\vid\riel1.mp4"

    cfg = load_config(config_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không mở được video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30
    wait_time = int(1000 / fps)

    print(f"Video FPS: {fps}")

    lane_memory = (None, None, None)
    timer_state = (False, 0.0)

    while True:
        start = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (cfg["FRAME_W"], cfg["FRAME_H"]))

        annotated, lane_info, raw_state, binary, lane_memory = detect_lane(
            frame, cfg, lane_memory
        )

        state, timer_state = junction_hold_logic(
            raw_state,
            timer_state,
            cfg["JUNCTION_COOLDOWN"]
        )

        cv2.putText(annotated, f"Direction: {lane_info}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(annotated, f"State: {state}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("Lane Detection", annotated)
        # cv2.imshow("Binary View", binary)

        process_time = int((time.time() - start) * 1000)
        actual_wait = max(1, wait_time - process_time)

        if cv2.waitKey(actual_wait) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
