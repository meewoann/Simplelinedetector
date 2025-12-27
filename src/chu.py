import cv2
import numpy as np

# ---------------- CONFIG ----------------
FRAME_W = 640
FRAME_H = 480
# ---------------------------------------
def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (0, height),
        (width, height),
        (width, int(height * 0.6)),
        (0, int(height * 0.6))
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask)


def detect_lane(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    cropped_edges = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=150
    )

    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x2 == x1:
                continue

            slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)

            if slope < 0:
                left_lines.append((slope, intercept))
            else:
                right_lines.append((slope, intercept))

            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Phân tích hướng
    if left_lines and right_lines:
        lane_info = "straight"
    elif left_lines:
        lane_info = "left"
    elif right_lines:
        lane_info = "right"
    else:
        lane_info = "no_lane"

    return frame, lane_info


def main():
    cap = cv2.VideoCapture(r"C:\Users\ADMIN\Desktop\osaka_MLS\line\vid\riel1.mp4")  # đổi sang video path nếu cần

    if not cap.isOpened():
        print("❌ Không mở được camera")
        return

    print("📷 Camera ready – nhấn Q để thoát")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        annotated, lane_info = detect_lane(frame)

        cv2.putText(
            annotated,
            f"Lane: {lane_info}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        print(f"lane_info: {lane_info}")

        cv2.imshow("Lane Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
