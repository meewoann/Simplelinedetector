import numpy as np
import cv2

class LaneDetection:
    def __init__(self, config_params):
        # Các tham số từ file config
        self.slices = config_params.get('slices', 20)
        self.width = config_params.get('width', 640)
        self.height = config_params.get('height', 480)
        self.step = -int(self.height * 0.4 / self.slices) # Khoảng cách giữa các slice
        self.bottom_row_index = self.height - 20
        self.top_row_index = self.bottom_row_index + (self.slices * self.step)
        
        # Ngưỡng cho Square Pulses
        self.min_height = config_params.get('min_height', 100)
        self.min_height_dif = config_params.get('min_height_dif', 30)
        self.pix_dif = config_params.get('pix_dif', 2)
        self.peaks_min_width = config_params.get('peaks_min_width', 5)
        self.max_allowed_dist = config_params.get('max_allowed_dist', 50)

    # --- PHẦN 1: POINTS DETECTION (SQUARE PULSES METHOD) ---
    
    def find_lane_peaks(self, slice_data):
        """
        Phát hiện các xung vuông (lane points) trong một lát cắt ngang của ảnh.
        """
        peaks = []
        inside_a_peak = False
        pix_num = 0
        
        # Duyệt qua từng pixel trong slice (histogram ngang)
        for i in range(self.pix_dif, len(slice_data) - self.pix_dif):
            pixel = slice_data[i]
            # Tính toán sự thay đổi cường độ để tìm cạnh xung
            height_dif_start = int(pixel) - int(slice_data[i - self.pix_dif])
            height_dif_end = int(pixel) - int(slice_data[i + self.pix_dif])

            if inside_a_peak:
                if height_dif_end > self.min_height_dif: # Kết thúc xung
                    inside_a_peak = False
                    if pix_num >= self.peaks_min_width: # Kiểm tra độ rộng xung
                        peak = i - (pix_num - self.pix_dif) // 2
                        peaks.append(peak)
                    pix_num = 0
                elif pixel > self.min_height: # Đang ở trong xung
                    pix_num += 1
                else:
                    inside_a_peak = False
            else:
                if pixel > self.min_height and height_dif_start > self.min_height_dif: # Bắt đầu xung
                    inside_a_peak = True
                    pix_num += 1
        return peaks

    def get_all_peaks(self, gray_frame):
        """
        Quét qua các lát cắt của ảnh để lấy toàn bộ các điểm lane tiềm năng.
        """
        all_lanes_points = []
        for height in range(self.bottom_row_index, self.top_row_index - 1, self.step):
            slice_data = gray_frame[height]
            peaks_in_slice = self.find_lane_peaks(slice_data)
            all_lanes_points = self.peaks_clustering(peaks_in_slice, height, all_lanes_points)
        return all_lanes_points

    # --- PHẦN 2: CLUSTERING INTO LANES (GOM CỤM LÀN ĐƯỜNG) ---

    def peaks_clustering(self, current_peaks, height, existing_lanes):
        """
        Gom các điểm vừa tìm được vào các làn đường đã có hoặc tạo làn mới.
        """
        if not existing_lanes:
            return [[[x, height]] for x in current_peaks]

        # dictionary để lưu vết quan hệ giữa điểm và làn
        lanes_dict = [{"point_index": -1, "dist": -1} for _ in range(len(existing_lanes))]
        points_used = [False for _ in range(len(current_peaks))]

        # Tìm điểm phù hợp nhất cho mỗi làn đường
        for l_idx, lane in enumerate(existing_lanes):
            best_dist = self.max_allowed_dist
            for p_idx, px in enumerate(current_peaks):
                if points_used[p_idx]: continue
                
                # Logic so sánh điểm với điểm cuối cùng của làn đường
                last_p = lane[-1]
                dist = abs(px - last_p[0])
                
                if dist < best_dist:
                    # Kiểm tra logic hướng đi (Expected Value)
                    is_valid, _ = self.verify_with_expected_value(lane, height, px)
                    if is_valid:
                        lanes_dict[l_idx] = {"point_index": p_idx, "dist": dist}
                        best_dist = dist

        # Cập nhật điểm vào làn hoặc tạo mới
        for l_idx, match in enumerate(lanes_dict):
            if match["point_index"] != -1:
                p_idx = match["point_index"]
                existing_lanes[l_idx].append([current_peaks[p_idx], height])
                points_used[p_idx] = True

        # Những điểm còn dư sẽ khởi tạo làn đường mới
        for p_idx, used in enumerate(points_used):
            if not used:
                existing_lanes.append([[current_peaks[p_idx], height]])

        return existing_lanes

    def verify_with_expected_value(self, lane, height, x_value):
        """
        Dự đoán vị trí tiếp theo dựa trên độ dốc của các điểm trước đó.
        """
        if len(lane) < 2:
            return True, 0
            
        # Tính độ dốc giữa điểm đầu và điểm cuối hiện tại của làn
        x_dif = lane[-1][0] - lane[0][0]
        y_dif = lane[-1][1] - lane[0][1]
        slope = x_dif / y_dif if y_dif != 0 else 0
        
        # Dự đoán X tại độ cao (height) mới
        expected_x = (height - lane[-1][1]) * slope + lane[-1][0]
        dist = abs(x_value - expected_x)
        
        return (dist < self.max_allowed_dist), dist

    def run(self, frame):
        """
        Hàm thực thi chính của module.
        """
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, self.min_height, 255, cv2.THRESH_BINARY)
        detected_lanes = self.get_all_peaks(thresh)
        
        # Lọc bỏ các làn có quá ít điểm (nhiễu)
        final_lanes = [l for l in detected_lanes if len(l) > 5]
        return final_lanes