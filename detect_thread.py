import os
import cv2
import time
import queue
import torch
import numpy as np
import threading
import traceback  # 新增：用來印出詳細錯誤
from datetime import datetime
from collections import deque

from utils.general import check_img_size, non_max_suppression_kpt, scale_coords, set_logging, increment_path
from utils.torch_utils import select_device, time_synchronized
from utils.plots import colors, plot_one_box
from utils.datasets import letterbox
from pathlib import Path

import pose_estimator   

CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.25
DRAW_SCALE = 0.3
DRAW_THICKNESS = 1
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

STOP_EVENT = threading.Event()



# 通用牆壁繪製函數
def draw_wall_lines(img, center, ceiling_start, left_wall_end, right_wall_end, color=WHITE, thickness=DRAW_THICKNESS):
    """
    繪製牆壁線條並返回座標點
    """
    # 繪製天花板垂直線
    cv2.line(img, ceiling_start, center, color, thickness)
    # 繪製左牆壁
    cv2.line(img, center, left_wall_end, color, thickness)
    # 繪製右牆壁
    cv2.line(img, center, right_wall_end, color, thickness)
    return center, left_wall_end, right_wall_end
   
# 用於獲取牆壁線上的座標點
def get_line_coordinates(start, end, list_pixels):

    # 計算直線斜率 m
    x1, y1 = start # 中心
    x2, y2 = end  # 牆壁
    m = (y2 - y1) / (x2 - x1)  # 斜率

    # # 定義一個空的列表來存儲直線上的所有像素點
    # line_pixels = []

    # 判斷 x1 是否小於 x2，確保遍歷順序
    if x1 > x2:
        x1, x2 = x2, x1  # 確保 x1 <= x2
        y1, y2 = y2, y1  # 同步交換 y1, y2

    # 遍歷 x 值
    for x in range(x1, x2 + 1):
        # 計算對應的 y 值
        y = int(m * (x - x1) + y1)
        # 儲存像素座標
        list_pixels.append((x, y))

    # print(list_pixels)
    return list_pixels

    
class DetectThread(threading.Thread):

    def __init__(self, cam_id, cam, model, model_lock, device, base_dir, fps, names, half, imgsz):
        super().__init__(daemon=True)

        self.current_alert_state = False
        self.stop_event = STOP_EVENT
        self.shared_frame_queue = queue.Queue(maxsize=10)
        self.cam_id = cam_id
        
        self.cam = cam
        self.model = model
        self.model_lock = model_lock
        self.device = device
        self.base_dir = Path(base_dir)

        self.fps = fps
        self.names = names
        self.half = half
        self.imgsz = imgsz

        self.stride = int(model.stride.max())
        
        # ==================== [新增] 校正與裁切參數設定 ====================
        self.enable_calibration = False
        self.calib_file = f'calibration_thermal_11.npz'  # 請確認路徑是否正確，或依 cam_id區分
        
        # 定義裁切範圍 (依照你的 cutBlack_resize.py 設定)
        # 格式: (x_start, x_end, y_start, y_end)
        self.crop_coords = (50, 550, 50, 420) 
        # 統一標準解析度
        self.standard_size = (640, 480)

        if os.path.exists(self.calib_file):
            try:
                # 1. 載入校正檔
                with np.load(self.calib_file) as X:
                    self.mtx = X['mtx']
                    self.dist = X['dist']
                
                # 建立映射表 (針對 640x480 的輸入)
                # 因為我們會先強制 Resize 成 640x480 才進來校正，所以這裡大小鎖定 (640, 480)
                self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                    self.mtx, self.dist, None, self.mtx, self.standard_size, cv2.CV_32FC1)
                self.enable_calibration = True
                print(f"[Cam{self.cam_id}] 校正模組啟用: 手動裁切範圍 {self.crop_coords}")

            except Exception as e:
                print(f"[Cam{self.cam_id}] 警告: 校正檔載入失敗，將使用原始影像推論。錯誤: {e}")
        else:
            print(f"警告：找不到 {self.calib_file}，將跳過校正步驟。")

        # ================================================================


        self.wall_config = None
        if self.cam_id == 0:
            self.wall_config = {'center': (90, 70), 'ceiling': (90, 0), 'left': (10, 120), 'right': (160, 110)}
        elif self.cam_id == 1:
            self.wall_config = {'center': (80, 70), 'ceiling': (80, 0), 'left': (10, 120), 'right': (160, 110)}
        else:
            self.wall_config = {'center': (105, 80), 'ceiling': (105, 0), 'left': (60, 120), 'right': (160, 90)}

        self.pose_estimator = pose_estimator.PoseEstimator(cf_thres=1, cam_id=cam_id, wall_config=self.wall_config)
        self.running = True
        self.prev_time = None
        self.fps_records = []
        self.sys_fps = fps
        self.frame_count = 0

        self.pre_event_seconds = 10
        self.post_event_seconds = 10
        self.b_recording = False
        self.event_start_time = 0 
        self.eventWriter = None
        
        # 初始化路徑變數
        self.final_cam_path = None
        self.temp_cam_path = None

        self.frame_buffer = deque(maxlen=int(self.sys_fps * self.pre_event_seconds))

    def run(self):
        # 加入全域 Try-Catch，防止執行緒無聲無息死亡
        try:
            while self.running:
                if self.stop_event.is_set():
                    break

                frame = self.cam.read()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # ==================== [新增] 校正 -> 裁切 -> 縮放 pipeline ====================
                if self.enable_calibration:
                    # 1. 校正 (Undistort)
                    # 使用 remap 比 undistort 快，因為 map 是預先算好的
                    frame_undist = cv2.remap(cv2.resize(frame, self.standard_size), self.mapx, self.mapy, cv2.INTER_LINEAR)
                    # 2. 裁切 (Crop) - 使用 numpy slicing
                    # Python Slice 語法是 [y_start : y_end, x_start : x_end]
                    x_s, x_e, y_s, y_e = self.crop_coords
                    frame_cropped = frame_undist[y_s:y_e, x_s:x_e]

                    # 3. 縮放 (Resize) - 回復到 640x480
                    frame = cv2.resize(frame_cropped, self.standard_size, interpolation=cv2.INTER_LINEAR)
                # ============================================================================

                self.prev_time = time.time()
                img0 = frame.copy()
                img = letterbox(img0, self.imgsz, stride=self.stride, auto=True)[0]
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()
                img /= 255.0
                img = img.unsqueeze(0)

                with self.model_lock:
                    pred = self.model(img, augment=False)[0]
                    pred = non_max_suppression_kpt(pred, conf_thres=CONF_THRESHOLD, iou_thres=IOU_THRESHOLD, classes=0, agnostic=False, kpt_label=True)

                im0 = img0.copy()

                if self.wall_config:

                    # 假設原始解析度是160x120，新的解析度是640x480
                    original_width, original_height = 160, 120  # 原始解析度 (可以根據你的需求調整)
                    new_width, new_height = self.standard_size  # 目標解析度 (來自目標大小，640x480)

                    # 計算縮放比例
                    scale_x = new_width / original_width
                    scale_y = new_height / original_height

                    # 縮放牆壁線座標
                    center_rescaled = (int(self.wall_config['center'][0] * scale_x), int(self.wall_config['center'][1] * scale_y))
                    ceiling_rescaled = (int(self.wall_config['ceiling'][0] * scale_x), int(self.wall_config['ceiling'][1] * scale_y))
                    left_rescaled = (int(self.wall_config['left'][0] * scale_x), int(self.wall_config['left'][1] * scale_y))
                    right_rescaled = (int(self.wall_config['right'][0] * scale_x), int(self.wall_config['right'][1] * scale_y))

                    center, left_wall_line, right_wall_line = draw_wall_lines(
                        im0, 
                        center_rescaled, 
                        ceiling_rescaled, 
                        left_rescaled, 
                        right_rescaled
                    )
                #     # 獲取牆壁線上的座標點 (如果需要)
                #     # left_wall_points = get_line_coordinates(center, left_wall_line, self.L_list_walls)
                #     # right_wall_points = get_line_coordinates(center, right_wall_line, self.R_list_walls)
                ##################################################################################

                # fps = calculate_fps(self.prev_time, self.frame_count)


                if len(pred[0]):
                    det = pred[0]
                    scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=True)
                    scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=True, step=3)

                    det = det[:1]
                    for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                        c = int(cls)
                        label = f'{self.names[c]} {conf:.2f}'
                        kpts = det[det_index, 6:]
                        kpts_list = [[round(kpts[j].item(), 2), round(kpts[j + 1].item(), 2), round(kpts[j + 2].item() * 2)] for j in range(0, len(kpts), 3)]

                        # 呼叫 Pose Estimator
                        _, R_vel, R_agl, avg_Ragl, L_vel, L_agl, avg_Lagl, hand_count, hand_alert_signal, near_ankle_state, leftAnkle_near_disL, rightAnkle_near_disL, leftAnkle_near_disR, rightAnkle_near_disR, nose_velocity, head_count, head_alert_signal, head_angle_change = self.pose_estimator.update_hand_info(kpts_list, self.frame_count)

                        # 繪圖
                        cv2.putText(im0, f"R Vel: {R_vel}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, DRAW_SCALE, WHITE, DRAW_THICKNESS, cv2.LINE_AA)
                        cv2.putText(im0, f"L Vel: {L_vel}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, DRAW_SCALE, WHITE, DRAW_THICKNESS, cv2.LINE_AA)
                        cv2.putText(im0, f"Hand Count: {hand_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, DRAW_SCALE, WHITE, DRAW_THICKNESS, cv2.LINE_AA)
                        cv2.putText(im0, "Hand Self Harm" if hand_alert_signal else " ", (80, 20), cv2.FONT_HERSHEY_SIMPLEX, DRAW_SCALE, GREEN, DRAW_THICKNESS, cv2.LINE_AA)

                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=2, kpt_label=True, kpts=kpts, steps=3, orig_shape=im0.shape[:2])

                        self.frame_buffer.append(im0.copy())

                        self.current_alert_state = hand_alert_signal or head_alert_signal

                        # ==================== 錄影邏輯 ====================
                        if hand_alert_signal or head_alert_signal:
                            self.event_start_time = time.time()

                            if not self.b_recording:
                                print(f'[Cam{self.cam_id}] 觸發事件！開始錄影...')
                                self.b_recording = True
                                
                                # 【修改 2】 動態生成當下的日期資料夾
                                now = datetime.now()
                                date_str = now.strftime("%Y/%m/%d")  # 例如 2025/12/29
                                timestamp = now.strftime("%Y%m%d_%H%M%S")

                                # 組合當日的完整路徑
                                current_save_dir = self.base_dir / date_str
                                if not current_save_dir.exists():
                                    current_save_dir.mkdir(parents=True, exist_ok=True)
                                    print(f"[系統] 跨日或新啟動，建立新資料夾: {current_save_dir}")

                                filename_base = f'Event_Cam{self.cam_id}_{timestamp}'
                                # 1. 最終檔名 (給 UI 看的)
                                self.final_cam_path = self.base_dir / f'{filename_base}.mp4'
                                # 2. 暫存檔名 (給 OpenCV 寫入用的，Server 會隱藏它)
                                # 重點：必須保留 .mp4 結尾，OpenCV 才會錄製！
                                self.temp_cam_path = str(self.base_dir / f'{filename_base}_TEMP.mp4')
                                
                                # self.temp_cam_path = str(self.final_cam_path) + ".temp"

                                h, w = im0.shape[:2]
                                record_fps = self.sys_fps if self.sys_fps > 0 else 10
                                # record_fps = int(max(self.sys_fps, 10))

                                print(f"[Cam{self.cam_id}] 建立錄影檔: {self.temp_cam_path} (FPS: {record_fps})")

                                # 確保路徑資料夾存在
                                if not os.path.exists(os.path.dirname(self.temp_cam_path)):
                                    print(f"警告：資料夾不存在，嘗試建立：{os.path.dirname(self.temp_cam_path)}")
                                    os.makedirs(os.path.dirname(self.temp_cam_path), exist_ok=True)

                                self.eventWriter = cv2.VideoWriter(self.temp_cam_path, cv2.VideoWriter_fourcc(*'mp4v'), record_fps, (w, h))
                                
                                # 寫入緩衝區影像
                                history_frames = list(self.frame_buffer)
                                for hist_frame in history_frames:
                                    if hist_frame is not None:
                                        self.eventWriter.write(hist_frame)

                        if self.b_recording and self.eventWriter is not None:
                            self.eventWriter.write(im0)

                            if (time.time() - self.event_start_time) > self.post_event_seconds: 
                                print(f"[Cam{self.cam_id}] 事件結束，停止錄影。")
                                self.stop_recording_and_save() # 呼叫存檔函式

                # 計算 FPS
                elapsed_time = time.time() - self.prev_time
                self.sys_fps = 1 / elapsed_time if elapsed_time > 0 else 0
                self.fps_records.append(self.sys_fps)

                if len(self.fps_records) == 50:
                    self.sys_fps = round(sum(self.fps_records) / len(self.fps_records))

                    safe_fps = min(self.sys_fps, 15) 

                    new_buffer_len = int(safe_fps * self.pre_event_seconds)

                    if new_buffer_len != self.frame_buffer.maxlen:
                        self.frame_buffer = deque(self.frame_buffer, maxlen=new_buffer_len)

                    self.pose_estimator.update_fps(self.sys_fps)
                   
                cv2.putText(im0, f'FPS: {round(self.sys_fps, 1)}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, DRAW_SCALE, WHITE, DRAW_THICKNESS, cv2.LINE_AA)

                # 推送影像
                if self.shared_frame_queue.full():
                    try: self.shared_frame_queue.get_nowait()
                    except queue.Empty: pass
                # self.shared_frame_queue.put(im0)
                self.shared_frame_queue.put((im0, self.current_alert_state))
                self.frame_count += 1
        
        except Exception as e:
            # 這是重點：如果發生錯誤，印出來，不要只卡住畫面
            print(f"!!! Cam{self.cam_id} DetectThread 發生嚴重錯誤 !!!")
            traceback.print_exc()
        
        finally:
            # 確保執行緒結束時（無論是正常停止還是崩潰），影片都能被儲存
            print(f"Cam{self.cam_id} 執行緒結束，執行清理...")
            self.stop_recording_and_save()

    # ==========================================
    # 獨立出來的存檔函式
    # ==========================================
    def stop_recording_and_save(self):
        self.b_recording = False
        if self.eventWriter is not None:
            self.eventWriter.release()
            self.eventWriter = None
            print(f"[Cam{self.cam_id}] 錄影結束，準備更名...") # Debug 訊息
            # 嘗試將 .temp 改名為 .mp4
            try:
                if self.temp_cam_path and os.path.exists(self.temp_cam_path):

                    if os.path.exists(self.final_cam_path):
                        os.remove(self.final_cam_path)

                    os.rename(self.temp_cam_path, str(self.final_cam_path))
                    print(f"影片救援成功/已儲存: {self.final_cam_path}")
                else:
                    print(f"錯誤：找不到暫存檔 {self.temp_cam_path}")
            except Exception as e:
                print(f"影片存檔失敗: {e}")

    # ==========================================
    # 重置 AI 狀態
    # ==========================================
    def reset_state(self):
        print(f"Cam{self.cam_id} 正在重置 AI 狀態...")
        self.frame_buffer.clear()
        if self.pose_estimator:
            self.pose_estimator.reset_memory()
        self.current_alert_state = False
    
    