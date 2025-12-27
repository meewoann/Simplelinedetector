import cv2
import numpy as np
from detect import LaneDetection 

def main():
    # 1. Cấu hình các tham số phát hiện
    config = {
        'width': 640,
        'height': 480,
        'slices': 50,
        'min_height': 140,
        'min_height_dif': 40,
        'peaks_min_width': 3,
        'max_allowed_dist': 30
    }

    # 2. Khởi tạo module
    ld = LaneDetection(config)

    # 3. Đường dẫn video input
    video_path = r"C:\Users\ADMIN\Desktop\osaka_MLS\line\vid\riel1.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Không thể mở file video.")
        return

    # ================== VIDEO OUTPUT ==================
    output_path = r"C:\Users\ADMIN\Desktop\osaka_MLS\line\vid\result1.mp4"

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # ổn định, dễ mở
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (config['width'], config['height'])
    )
    # ==================================================

    print("Đang xử lý video... Nhấn 'q' để thoát.")

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255)
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (config['width'], config['height']))
        
        lanes = ld.run(frame)

        # Vẽ vùng slice
        cv2.line(frame, (0, ld.bottom_row_index),
                 (config['width'], ld.bottom_row_index),
                 (100, 100, 100), 1)
        cv2.line(frame, (0, ld.top_row_index),
                 (config['width'], ld.top_row_index),
                 (100, 100, 100), 1)

        for i, lane in enumerate(lanes):
            color = colors[i % len(colors)]
            for point in lane:
                cv2.circle(frame, (point[0], point[1]), 3, color, -1)

            if len(lane) > 1:
                pts = np.array(lane, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, color, 2)

        # cv2.putText(
        #     frame,
        #     f"Lanes detected: {len(lanes)}",
        #     (20, 40),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,
        #     (255, 255, 255),
        #     2
        # )

        # ========== GHI FRAME RA VIDEO ==========
        out.write(frame)
        # =======================================

        cv2.imshow("Lane Detection - Result", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()          # QUAN TRỌNG
    cv2.destroyAllWindows()

    print(f"Done. Video đã lưu tại:\n{output_path}")

if __name__ == "__main__":
    main()
