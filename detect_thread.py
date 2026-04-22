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


# 是否寫成CSV
# RECORDING_DIAGRAM_TOCSV = True

# CONF_THRESHOLD = 0.45
# IOU_THRESHOLD = 0.25
# DRAW_SCALE = 0.2
# DRAW_THICKNESS = 1
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)
# GREEN = (0, 255, 0)

STOP_EVENT = threading.Event()


# 通用牆壁繪製函數
def draw_wall_lines(img, center, ceiling_start, left_wall_end, right_wall_end, color=(255,255,255), thickness=1):
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


class VideoRecordsThread(threading.Thread):
    '''
        負責在背景獨立將 Queue 中的影像寫入硬碟，
        避免龐大的I/O 操作阻塞主推論與UI執行緒。
    '''
    def __init__(self, filepath, fps, size, pre_frames, record_queue, cam_id):
        super().__init__(daemon=True)
        self.filepath = filepath
        self.fps = fps
        self.size = size
        self.pre_frames = pre_frames
        self.record_queue = record_queue
        self.cam_id = cam_id

    def run(self):
        writer = None
        try:
            writer = cv2.VideoWriter(self.filepath, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)
            # 優先寫入預錄緩衝區(Pre-event frames)
            for frame in self.pre_frames:
                writer.write(frame)

            while True:
                frame = self.record_queue.get()
                if frame is None:  # 收到終止信號
                    break   
                writer.write(frame)
            print(f'[Cam{self.cam_id}] 影片已於背景錄製完成並儲存: {self.filepath}')
        except Exception as e:
            print(f'!!! [Cam{self.cam_id}] 背影錄影發生錯誤: {e} !!!')
            traceback.print_exc()
        finally:
            if writer is not None:
                writer.release()


class DetectThread(threading.Thread):

    def __init__(self, cam_id, cam, model, model_lock, device, base_dir, fps, names, half, imgsz, config=None):
        super().__init__(daemon=True)
        self.cfg = config if config else {}

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
        
        # ***** 從設定檔載入參數 ******
        # 預測的門檻
        det_cfg = self.cfg.get('detection_params', {})
        yolo_cfg = det_cfg.get('yolo', {})
        self.conf_threshold = yolo_cfg.get('conf_threshold', 0.45)
        self.iou_threshold = yolo_cfg.get('iou_threshold', 0.25)
        # 錄製的時間長短與是否輸出 CSV
        rec_cfg = det_cfg.get('recording', {})
        self.pre_event_seconds = rec_cfg.get('pre_event_seconds', 10)
        self.post_event_seconds = rec_cfg.get('post_event_seconds', 10)
        self.record_diagram_to_csv = det_cfg.get('record_diagram_to_csv', False)
        # 繪圖參數 
        draw_cfg = self.cfg.get('drawing_params', {})
        self.base_font_scale = draw_cfg.get('font_scale', 0.2)
        self.base_line_thickness = draw_cfg.get('line_thickness', 1)
        colors_cfg = draw_cfg.get('colors', {})
        self.white = tuple(colors_cfg.get('white', [255, 255, 255]))
        self.green = tuple(colors_cfg.get('green', [0, 255, 0]))
        # 校正與裁切參數
        self.enable_calibration = False
        self.calib_file = det_cfg.get('calibration_file', 'calibration_thermal_11.npz')
        self.crop_coords = tuple(det_cfg.get('crop_coords', [50, 550, 50, 420]))
        self.standard_size = (640, 480)

        # ==================== [新增] 校正與裁切參數設定 ====================
        # self.enable_calibration = False
        # self.calib_file = f'calibration_thermal_11.npz'  # 請確認路徑是否正確，或依 cam_id區分
        
        # # 定義裁切範圍 (依照你的 cutBlack_resize.py 設定)
        # # 格式: (x_start, x_end, y_start, y_end)
        # self.crop_coords = (50, 550, 50, 420) 
        # # 統一標準解析度
        # self.standard_size = (640, 480)

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
        # 原始解析度是160x120，校正後的解析度是640x480
        original_width, original_height = 160, 120  # 原始解析度 (可以根據你的需求調整)
        new_width, new_height = self.standard_size  # 目標解析度 (來自目標大小，640x480)
        # 計算縮放比例
        self.scale_x = new_width / original_width
        self.scale_y = new_height / original_height
        # 根據縮放比例調整文字大小
        self.scaled_text_size = self.base_font_scale * self.scale_x  # 這裡調整文字大小
        self.scaled_thickness = int(self.base_line_thickness)  # 根據比例調整文字粗細
        print(f'繪製的大小{self.scaled_text_size}, 繪製的粗細{self.scaled_thickness}')
        
        # self.raw_wall_config = None 
        # 動態牆壁線設定
        wall_cfgs = self.cfg.get('wall_configs', {})
        cam_key = str(self.cam_id)
        if cam_key in wall_cfgs:
            raw_wall = wall_cfgs[cam_key]
            print(f"👉 [Cam{self.cam_id}] 成功讀取 JSON 牆壁設定: {raw_wall['center']}") # 加入這行        
        else:
            raw_wall = wall_cfgs.get('default', {'center': [105, 80], 'ceiling': [105, 0], 'left': [60, 120], 'right': [160, 90]})
            print(f"⚠️ [Cam{self.cam_id}] 警告：讀不到設定，啟用預設牆壁座標: {raw_wall['center']}") # 加入這行
        # # 拍向深處的鏡頭
        # if self.cam_id == 0:
        #     self.raw_wall_config = {'center': (90, 70), 
        #                             'ceiling': (90, 0), 
        #                             'left': (10, 120), 
        #                             'right': (160, 110)}
        # # 拍向門口的鏡頭
        # elif self.cam_id == 1:
        #     self.raw_wall_config = {'center': (80, 70), 
        #                             'ceiling': (80, 0), 
        #                             'left': (10, 120), 
        #                             'right': (160, 110)}
        # else:
        #     self.raw_wall_config = {'center': (105, 80), 
        #                             'ceiling': (105, 0), 
        #                             'left': (60, 120), 
        #                             'right': (160, 90)}

        self.raw_wall_config = {
            'center': tuple(raw_wall['center']),
            'ceiling': tuple(raw_wall['ceiling']),
            'left' : tuple(raw_wall['left']),
            'right': tuple(raw_wall['right'])
        }
        
        # 建立一個 "縮放後" 的 Config 給 PoseEstimator 用
        self.dict_scaled_wall_config = {}
        if self.raw_wall_config:
            self.dict_scaled_wall_config['center'] = (int(self.raw_wall_config['center'][0] * self.scale_x), int(self.raw_wall_config['center'][1] * self.scale_y))
            self.dict_scaled_wall_config['ceiling'] = (int(self.raw_wall_config['ceiling'][0] * self.scale_x), int(self.raw_wall_config['ceiling'][1] * self.scale_y))
            self.dict_scaled_wall_config['left'] = (int(self.raw_wall_config['left'][0] * self.scale_x), int(self.raw_wall_config['left'][1] * self.scale_y))
            self.dict_scaled_wall_config['right'] = (int(self.raw_wall_config['right'][0] * self.scale_x), int(self.raw_wall_config['right'][1] * self.scale_y))
        # ================================================================
        self.pose_estimator = pose_estimator.PoseEstimator(cf_thres=1, 
                                                           cam_id=cam_id, 
                                                           scale_ratio=self.scale_x,
                                                           wall_config=self.dict_scaled_wall_config,
                                                           record_diagram_to_csv=self.record_diagram_to_csv,
                                                           algo_cfg = self.cfg.get('algorithm_thresholds', {},
                                                                                   )
                                                           )  # <---【新增】傳入縮放倍率 (約為 4.0))
        self.running = True
        self.prev_time = None
        # 改成序列的方式
        self.fps_records = deque(maxlen=50)  # 使用 deque 確保只保留最新 50 筆，自動滾動
        self.loaded_fps = fps                # 新增：專門儲存「有偵測到人時」的穩定 FPS，預設為系統給定的 fps
        self.sys_fps = fps
        self.frame_count = 0

        # ==================== [新增] FPS 延遲校正狀態機 ====================
        # 若在校正完成前就觸發錄影，將使用啟動時傳入的保守 fps 當作備用基準 (來源：合理推論，避免除以零)
        self.calibrated_fps = fps if fps > 0 else 15 
        self.is_fps_calibrated = False   # 標記是否已經完成有人狀態的 FPS 校正
        self.load_fps_buffer = []        # 用來收集 30 幀的單幀耗時
        # ===================================================================

        # self.pre_event_seconds = 10
        # self.post_event_seconds = 10
        self.b_recording = False
        self.event_start_time = 0 
        # self.eventWriter = None
        # 實例化背景錄影類別 *************
        self.record_thread = None
        self.record_queue = queue.Queue()
        # *******************************
        # 初始化路徑變數
        self.final_cam_path = None
        self.temp_cam_path = None

        self.frame_buffer = deque(maxlen=int(self.sys_fps * self.pre_event_seconds))


        # ==================== [新增] 初始化 24 小時錄影執行緒 ====================
        self.continuous_thread = ContinuousRecordThread(
            base_dir=self.base_dir,
            cam_id=self.cam_id,
            fps=self.sys_fps,
            size=self.standard_size
        )
        self.continuous_thread.start()
        # =========================================================================




    def run(self):
        # 加入全域 Try-Catch，防止執行緒無聲無息死亡
        try:
            self.prev_time = time.time()
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

                # self.prev_time = time.time()
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
                    pred = non_max_suppression_kpt(pred, conf_thres=self.conf_threshold, iou_thres=self.iou_threshold, classes=0, agnostic=False, kpt_label=True)

                im0 = img0.copy()

                if self.dict_scaled_wall_config:

                    # 縮放牆壁線座標
                    center_rescaled = (self.dict_scaled_wall_config['center'][0] , self.dict_scaled_wall_config['center'][1])
                    ceiling_rescaled = (self.dict_scaled_wall_config['ceiling'][0], self.dict_scaled_wall_config['ceiling'][1])
                    left_rescaled = (self.dict_scaled_wall_config['left'][0], self.dict_scaled_wall_config['left'][1])
                    right_rescaled = (self.dict_scaled_wall_config['right'][0], self.dict_scaled_wall_config['right'][1])

                    center, left_wall_line, right_wall_line = draw_wall_lines(
                        im0, 
                        center_rescaled, 
                        ceiling_rescaled, 
                        left_rescaled, 
                        right_rescaled,
                        color=self.white, 
                        thickness=self.scaled_thickness
                    )
                    # 獲取牆壁線上的座標點 (如果需要)
                    # left_wall_points = get_line_coordinates(center, left_wall_line, self.L_list_walls)
                    # right_wall_points = get_line_coordinates(center, right_wall_line, self.R_list_walls)
                ##################################################################################

                # 計算 FPS =======================================================================
                current_time = time.time()
                elapsed_time = current_time - self.prev_time
                self.prev_time = current_time     # 更新 prev_time 給下一幀使用
                current_frame_fps = 1 / elapsed_time if elapsed_time > 0 else 0
                # self.sys_fps = 1 / elapsed_time if elapsed_time > 0 else 0
                self.sys_fps = current_frame_fps

                # self.fps_records.append(self.sys_fps)

                # # 只保留最近 50 筆 計算平均FPS，當作序列長度
                # if len(self.fps_records) == 50:
                #     self.sys_fps = round(sum(self.fps_records) / len(self.fps_records))
                #     # 加入合理範圍限制 (例如限制在 5 ~ 30 之間)
                #     safe_fps = int(max(5, min(self.sys_fps, 30)))
                #     # ******************************************
                #     new_buffer_len = int(safe_fps * self.pre_event_seconds)
                #     if new_buffer_len != self.frame_buffer.maxlen:
                #         self.frame_buffer = deque(self.frame_buffer, maxlen=new_buffer_len)

                #     self.pose_estimator.update_fps(self.sys_fps)
                    
                # cv2.putText(im0, f'FPS: {round(self.sys_fps, 1)}', (int(10 * self.scale_x), int(10 * self.scale_y)), cv2.FONT_HERSHEY_SIMPLEX, self.scaled_text_size, WHITE, self.scaled_thickness, cv2.LINE_AA)
                # 在影像上顯示 FPS，這部分也應該根據縮放比例來調整
                cv2.putText(im0, f'FPS: {round(current_frame_fps, 1)}', 
                            (int(10 * self.scale_x), int(100 * self.scale_y)),  # 調整位置
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            self.scaled_text_size,  # 根據縮放比例調整字體大小
                            self.white, 
                            self.scaled_thickness,  # 根據縮放比例調整粗細
                            cv2.LINE_AA)
                
                if len(pred[0]):
                    # 【核心修改】偵測到人時的延遲校正機制
                    if not self.is_fps_calibrated:
                        self.load_fps_buffer.append(elapsed_time)
                        
                        # 收集滿 30 幀 (約 2~3 秒) 後進行結算與鎖定
                        if len(self.load_fps_buffer) >= 50:
                            avg_load_time = sum(self.load_fps_buffer) / 50.0
                            # 防呆限制：鎖定在 5 ~ 30 FPS 之間
                            self.calibrated_fps = int(max(5, min(1 / avg_load_time if avg_load_time > 0 else 15, 30)))
                            self.is_fps_calibrated = True # 鎖定，後續不再計算
                            
                            print(f"[Cam{self.cam_id}] 系統負載 FPS 校正完成，鎖定為: {self.calibrated_fps} FPS")
                    # # *************************************************************************
                    # # 【新增核心邏輯】只有偵測到人，才將當下 FPS 加入陣列計算
                    # self.fps_records.append(self.sys_fps)

                    # # 只要收集到 10 幀「有人的畫面」，就開始動態更新平均 FPS，反應速度會比等 50 幀更快
                    # if len(self.fps_records) >= 10:
                    #     self.loaded_fps = sum(self.fps_records) / len(self.fps_records)
                    #     safe_fps = int(max(5, min(self.loaded_fps, 30)))
                        
                            # 動態調整錄影前緩衝區 (pre_event_seconds) 大小
                            new_buffer_len = int(self.calibrated_fps * self.pre_event_seconds)
                            if new_buffer_len != self.frame_buffer.maxlen:
                                # deque 重新賦值時，若新長度較短，會自動保留最新的畫面，丟棄最舊的畫面
                                self.frame_buffer = deque(self.frame_buffer, maxlen=new_buffer_len)

                            # 更新給 Pose Estimator 的時序參數
                            self.pose_estimator.update_fps(self.calibrated_fps)

                    # *************************************************************************

                    det = pred[0]
                    scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=True)
                    scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=True, step=3)
                    
                    det = det[:1]  # 只處理最高信心度的偵測結果
                    for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                        c = int(cls)
                        label = f'{self.names[c]} {conf:.2f}'
                        kpts = det[det_index, 6:]
                        kpts_list = [[round(kpts[j].item(), 2), round(kpts[j + 1].item(), 2), round(kpts[j + 2].item() * 2)] for j in range(0, len(kpts), 3)]

                        # 呼叫 Pose Estimator
                        _, R_vel, R_agl, avg_Ragl, L_vel, L_agl, avg_Lagl, hand_count, hand_alert_signal, near_ankle_state, leftAnkle_near_disL, rightAnkle_near_disL, leftAnkle_near_disR, rightAnkle_near_disR, nose_velocity, head_count, head_alert_signal, head_angle_change = self.pose_estimator.update_hand_info(kpts_list, self.frame_count)
                        # 調整文字顯示的位置，因為影像被裁切放大，需要調整
                        cv2.putText(im0, f"R Vel: {R_vel}", (int(10 * self.scale_x), int(10 * self.scale_y)), cv2.FONT_HERSHEY_SIMPLEX, self.scaled_text_size, self.white, self.scaled_thickness, cv2.LINE_AA)
                        cv2.putText(im0, f"L Vel: {L_vel}", (int(10 * self.scale_x), int(20 * self.scale_y)), cv2.FONT_HERSHEY_SIMPLEX, self.scaled_text_size, self.white, self.scaled_thickness, cv2.LINE_AA)
                        cv2.putText(im0, f'Cam: {self.cam_id}', (int(80 * self.scale_x), int(10 * self.scale_y)), cv2.FONT_HERSHEY_SIMPLEX, self.scaled_text_size, self.white, self.scaled_thickness, cv2.LINE_AA)
                        cv2.putText(im0, f"Hand Count: {hand_count}", (int(10 * self.scale_x), int(30 * self.scale_y)), cv2.FONT_HERSHEY_SIMPLEX, self.scaled_text_size, self.white, self.scaled_thickness, cv2.LINE_AA)
                        cv2.putText(im0, "Hand Self Harm" if hand_alert_signal else " ", (int(80 * self.scale_x), int(20 * self.scale_y)), cv2.FONT_HERSHEY_SIMPLEX, self.scaled_text_size, self.green, self.scaled_thickness* 2, cv2.LINE_AA)
                        cv2.putText(im0, "near wall" if near_ankle_state else " ", (int(80 * self.scale_x), int(30 * self.scale_y)), cv2.FONT_HERSHEY_SIMPLEX, self.scaled_text_size, self.green, self.scaled_thickness * 2, cv2.LINE_AA)
                        cv2.putText(im0, f"Head Vel: {nose_velocity}", (int(10 * self.scale_x), int(40 * self.scale_y)), cv2.FONT_HERSHEY_SIMPLEX, self.scaled_text_size, self.white, self.scaled_thickness, cv2.LINE_AA)
                        cv2.putText(im0, f"Head Count: {head_count}", (int(10 * self.scale_x), int(50 * self.scale_y)), cv2.FONT_HERSHEY_SIMPLEX, self.scaled_text_size, self.white, self.scaled_thickness, cv2.LINE_AA)
                        cv2.putText(im0, "Head Self Harm" if head_alert_signal else " ", (int(80 * self.scale_x), int(40 * self.scale_y)), cv2.FONT_HERSHEY_SIMPLEX, self.scaled_text_size, self.green, self.scaled_thickness* 2, cv2.LINE_AA)
        
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=2, kpt_label=True, kpts=kpts, steps=3, orig_shape=im0.shape[:2])

                        # self.frame_buffer.append(im0.copy())
                        # self.current_alert_state = hand_alert_signal or head_alert_signal
                        # 在UI上細分自傷狀態
                        alert_types = []
                        if hand_alert_signal:
                            alert_types.append("發生手部自傷行為")
                            # print("DEBUG: 偵測到手部異常！")  # <--- 建議打開註解測試
                        elif head_alert_signal:
                            alert_types.append("發生頭部自傷行為")
                            # print("DEBUG: 偵測到頭部異常！")  # <--- 建議打開註解測試
                        
                        # 如果有任何警報，組合成字串 (例如 "手部" 或 "手部 & 頭部")；否則為 None
                        if len(alert_types) > 0:
                            self.current_alert_state = " & ".join(alert_types)
                        else:
                            self.current_alert_state = None
                        # ==================== 錄影邏輯 ====================
                        if hand_alert_signal or head_alert_signal:
                            self.event_start_time = time.time()

                            if not self.b_recording:
                                # print(f'[Cam{self.cam_id}] 觸發事件！開始錄影...')
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
                                self.final_cam_path = str(current_save_dir / f'{filename_base}.mp4')
                                
                                h, w = im0.shape[:2]
                                record_fps = self.calibrated_fps

                                # 確保路徑資料夾存在
                                if not os.path.exists(os.path.dirname(self.final_cam_path)):
                                    print(f"警告：資料夾不存在，嘗試建立：{os.path.dirname(self.final_cam_path)}")
                                    os.makedirs(os.path.dirname(self.final_cam_path), exist_ok=True)
                                
                                history_frames = list(self.frame_buffer)
                                self.record_queue = queue.Queue()

                                self.record_thread = VideoRecordsThread(
                                    filepath=self.final_cam_path, 
                                    fps=record_fps, 
                                    size=(w, h), 
                                    pre_frames=history_frames, 
                                    record_queue=self.record_queue, 
                                    cam_id=self.cam_id
                                    )
                                self.record_thread.start()
                                print(f'[Cam{self.cam_id}] 背景錄影執行緒啟動 (FPS: {record_fps})')
                                
                                # self.eventWriter = cv2.VideoWriter(self.final_cam_path, cv2.VideoWriter_fourcc(*'mp4v'), record_fps, (w, h))
                                # print(f'儲存成影片的FPS為: {record_fps}')
                                # 寫入緩衝區影像
                                # history_frames = list(self.frame_buffer)
                                # for hist_frame in history_frames:
                                #     if hist_frame is not None:
                                #         self.eventWriter.write(hist_frame)

                # 有骨幹的影像
                # self.frame_buffer.append(im0.copy())
                # 無骨幹的影像
                self.frame_buffer.append(img0.copy())
                
                
                # 寫入 24 小時連續錄影 (若佇列未滿)
                if not self.continuous_thread.record_queue.full():
                    self.continuous_thread.record_queue.put(img0.copy())

                if self.b_recording and self.record_thread is not None:
                    #     self.record_queue.put(im0.copy())
                    self.record_queue.put(img0.copy())
                    if (time.time() - self.event_start_time) > self.post_event_seconds: 
                        self.stop_recording_and_save()

                # 推送影像
                if self.shared_frame_queue.full():
                    try: 
                        self.shared_frame_queue.get_nowait()
                    except queue.Empty: 
                        pass
                self.shared_frame_queue.put((im0, self.current_alert_state))
                self.frame_count += 1
        
        except Exception as e:
            print(f"!!! Cam{self.cam_id} DetectThread 發生嚴重錯誤 !!!")
            traceback.print_exc()
        
        finally:
            # 確保執行緒結束時（無論是正常停止還是崩潰），影片都能被儲存
            # print(f"Cam{self.cam_id} 影片儲存執行緒結束，執行清理...")
            self.stop_recording_and_save()
            # ==================== [新增] 安全停止 24H 錄影 ====================
            if hasattr(self, 'continuous_thread') and self.continuous_thread is not None:
                self.continuous_thread.stop()
                self.continuous_thread.join(timeout=2.0)
            # ==================================================================


    # ==========================================
    # 獨立出來的存檔函式
    # ==========================================
    def stop_recording_and_save(self):
        self.b_recording = False
        if getattr(self, 'record_thread', None) is not None:
            self.record_queue.put(None)  # 放進None 作為結束標記
            self.recorder_thread = None  # 解除參照，讓下次可順利建立執行緒
            print(f'[Cam{self.cam_id}] 偵測事件結束，已發送背景寫入結束信號。')
        # if self.eventWriter is not None:
        #     self.eventWriter.release()
        #     self.eventWriter = None
        #     print(f"[Cam{self.cam_id}] 影片已錄製完成並儲存: {self.final_cam_path}")            

    # ==========================================
    # 重置 AI 狀態
    # ==========================================
    def reset_state(self):
        # print(f"Cam{self.cam_id} 正在重置 AI 狀態...")
        self.frame_buffer.clear()
        if self.pose_estimator:
            self.pose_estimator.reset_memory()
        self.current_alert_state = False
    
    
class ContinuousRecordThread(threading.Thread):
    '''
    負責在背景獨立進行 24 小時全天候錄影，
    自動依據「小時」切割影片檔案，避免單一檔案過大導致毀損。
    '''
    def __init__(self, base_dir, cam_id, fps, size):
        super().__init__(daemon=True)
        self.base_dir = Path(base_dir)
        self.cam_id = cam_id
        # 若系統初始 FPS 為 0 或異常，給予預設值 15 (來源：detect_thread.py 邏輯推論)
        self.fps = fps if fps > 0 else 15 
        self.size = size
        self.record_queue = queue.Queue(maxsize=300) # 緩衝區避免 I/O 阻塞
        self.running = True
        self.writer = None
        self.current_hour = None

    def _get_writer(self):
        now = datetime.now()
        # 以小時為單位命名，例如：20251226_11
        hour_str = now.strftime("%Y%m%d_%H")
        
        if self.current_hour != hour_str:
            if self.writer is not None:
                self.writer.release()
            
            self.current_hour = hour_str
            # 建立獨立的 Continuous 資料夾區分事件錄影
            save_dir = self.base_dir / "Continuous" / now.strftime("%Y/%m/%d")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = save_dir / f"Cam{self.cam_id}_Continuous_{hour_str}.mp4"
            self.writer = cv2.VideoWriter(str(filepath), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)
            print(f'[Cam{self.cam_id}] 建立連續錄影新檔案: {filepath}')
            
        return self.writer

    def run(self):
        try:
            while self.running:
                try:
                    # 使用 timeout 確保能響應 running 狀態改變
                    frame = self.record_queue.get(timeout=1)
                    if frame is None: # 收到終止信號
                        break
                    
                    writer = self._get_writer()
                    writer.write(frame)
                except queue.Empty:
                    continue
        except Exception as e:
            print(f'!!! [Cam{self.cam_id}] 24小時錄影發生錯誤: {e} !!!')
            traceback.print_exc()
        finally:
            if self.writer is not None:
                self.writer.release()

    def stop(self):
        self.running = False
        self.record_queue.put(None)