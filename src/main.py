import cv2
import numpy as np

# ------------------ CONFIG ------------------
FRAME_W = 640
FRAME_H = 480
MAX_STEER_ANGLE = 45.0  # độ
# --------------------------------------------

roi_box = None  # (x, y, w, h)


def region_of_interest(img, roi_box):
    x, y, w, h = roi_box
    mask = np.zeros_like(img)
    mask[y:y + h, x:x + w] = 255
    return cv2.bitwise_and(img, mask)


def compute_steering_angle(frame, edges, roi_box):
    roi = region_of_interest(edges, roi_box)

    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=50
    )

    left_pts, right_pts = [], []
    vis = frame.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)

            if slope < -0.5 and x1 < FRAME_W // 2:
                left_pts += [(x1, y1), (x2, y2)]
            elif slope > 0.5 and x1 > FRAME_W // 2:
                right_pts += [(x1, y1), (x2, y2)]

            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    steering_angle = 0.0

    if left_pts and right_pts:
        left_fit = np.polyfit(
            [p[1] for p in left_pts],
            [p[0] for p in left_pts],
            1
        )
        right_fit = np.polyfit(
            [p[1] for p in right_pts],
            [p[0] for p in right_pts],
            1
        )

        y_eval = FRAME_H
        left_x = np.polyval(left_fit, y_eval)
        right_x = np.polyval(right_fit, y_eval)

        lane_center = (left_x + right_x) / 2
        frame_center = FRAME_W / 2
        offset = lane_center - frame_center

        steering_angle = -offset / frame_center * MAX_STEER_ANGLE

        # Vẽ vùng lane
        pts = np.array([
            (int(left_x), y_eval),
            (int(np.polyval(left_fit, y_eval - 100)), y_eval - 100),
            (int(np.polyval(right_fit, y_eval - 100)), y_eval - 100),
            (int(right_x), y_eval)
        ], np.int32)

        cv2.fillPoly(vis, [pts.reshape((-1, 1, 2))], (255, 200, 0))

    return vis, steering_angle


def main():
    global roi_box

    cap = cv2.VideoCapture(r"C:\Users\ADMIN\Desktop\osaka_MLS\line\vid\riel1.mp4")  # đổi sang video path nếu cần
    if not cap.isOpened():
        print("❌ Không mở được camera")
        return

    print("📷 Camera ready")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        if roi_box is None:
            roi_box = cv2.selectROI("Select ROI", frame, False, False)
            cv2.destroyWindow("Select ROI")
            print("✅ ROI:", roi_box)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        vis, angle = compute_steering_angle(frame, edges, roi_box)

        cv2.putText(
            vis,
            f"Steering angle: {angle:.2f} deg",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.imshow("Lane Detection", vis)
        print(f"Steering angle: {angle:.2f}°")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
