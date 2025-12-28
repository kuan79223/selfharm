import sys
import cv2
import time
import queue
import torch
import struct
import socket
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque

# PyQt6 引用
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QWidget, 
    QVBoxLayout, QHBoxLayout, QGridLayout, 
    QTextEdit, QGroupBox
)
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer, pyqtSlot

# YOLOv7 引用
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt, scale_coords, increment_path
from utils.plots import colors, plot_one_box
from utils.datasets import letterbox
import pose_estimator

# ============ [核心設定] ============
# RPI_CONFIGS = [
    # {'ip': '192.168.50.228', 'port': 8001, 'mapped_id': 0}, 
    # {'ip': '192.168.50.43',  'port': 8002, 'mapped_id': 1} 
# ]
RPI_CONFIGS = [
    {'ip': '192.168.137.98', 'port': 8001, 'mapped_id': 0}
    # {'ip': '192.168.50.43',  'port': 8002, 'mapped_id': 1} 
]

CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.25
IMG_SIZE = 640

# 共用變數
latest_frames = {}           
frame_lock = threading.Lock()
STOP_EVENT = threading.Event()


# ============ [2] VideoSaver (維持原樣) ============
class VideoSaver(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.queue = queue.Queue()
        self.active = True
        self.writers = {} 

    def write_frame(self, path, frame, fps, width, height):
        if self.active:
            self.queue.put((path, frame.copy(), fps, width, height))

    def run(self):
        while self.active or not self.queue.empty():
            try:
                task = self.queue.get(timeout=1)
                path, frame, fps, w, h = task
                if path not in self.writers:
                    self.writers[path] = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                self.writers[path].write(frame)
            except queue.Empty: continue
            except Exception as e: print(f"[VideoSaver] Error: {e}")
        for w in self.writers.values(): w.release()

    def stop(self): self.active = False

video_saver = VideoSaver()

# ============ [3] StreamReceiver (維持原樣) ============
class StreamReceiver(threading.Thread):
    def __init__(self, ip, port, mapped_id):
        super().__init__(daemon=True)
        self.ip = ip
        self.port = port
        self.mapped_id = mapped_id 
        self.index_name = f"Rx-{ip.split('.')[-1]}"

    def recvall(self, sock, n):
        data = b''
        while len(data) < n:
            try:
                packet = sock.recv(n - len(data))
                if not packet: return None
                data += packet
            except: return None
        return data

    def run(self):
        global latest_frames
        
        while not STOP_EVENT.is_set():
            client = None
            try:
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.settimeout(5) 
                client.connect((self.ip, self.port))
                client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
                
                payload_fmt = "<BI"
                payload_size = struct.calcsize(payload_fmt)

                while not STOP_EVENT.is_set():
                    header = self.recvall(client, payload_size)
                    if not header: break
                    _, msg_size = struct.unpack(payload_fmt, header)
                    frame_data = self.recvall(client, msg_size)
                    if not frame_data: break
                    
                    np_data = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

                    if frame is not None:
                        with frame_lock:
                            latest_frames[self.mapped_id] = frame
                            
            except (ConnectionRefusedError, socket.timeout, OSError):
                time.sleep(2)
            except Exception as e:
                time.sleep(2)
            finally:
                if client: client.close()

# ============ [4] DetectThread (修改為發送訊號) ============
class DetectThread(threading.Thread):
    def __init__(self, cam_id, model, model_lock, device, save_dir, names, half, imgsz):
        super().__init__(daemon=True)
        self.cam_id = cam_id 
        self.model = model
        self.model_lock = model_lock
        self.device = device
        self.save_dir = save_dir
        self.names = names
        self.half = half
        self.imgsz = imgsz
        
        # 牆壁參數
        self.wall_config = {'center': (105, 80), 'ceiling': (105, 0), 'left': (60, 120), 'right': (160, 90)}
        if self.cam_id == 0: 
             self.wall_config = {'center': (90, 70), 'ceiling': (90, 0), 'left': (10, 120), 'right': (160, 110)}

        self.wrist_shift = pose_estimator.PoseEstimator(cf_thres=1, cam_id=cam_id, wall_config=self.wall_config)
        self.video_procFile = str(self.save_dir / f'Record_Cam{self.cam_id}_{datetime.now().strftime("%H%M%S")}.mp4')
        
        # FPS 與變數
        self.prev_time = None
        self.sys_fps = 0     
        self.rec_fps = 10    
        self.fps_records = []
        self.fps_initialized = False 
        
        self.preEvent_queue = queue.Queue(maxsize=10*8)
        self.is_event_recording = False
        self.event_start_time = 0
        self.event_filename = ""

    def run(self):
        
        while not STOP_EVENT.is_set():
            # 1. 取圖
            frame = None
            with frame_lock:
                if self.cam_id in latest_frames:
                    frame = latest_frames[self.cam_id].copy()
            
            if frame is None:
                time.sleep(0.01)
                continue

            self.prev_time = time.time() # 開始計時
            im0 = frame.copy()
            h, w = im0.shape[:2]
            
            # YOLO 推論
            img = letterbox(im0, self.imgsz, stride=64, auto=True)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()
            img /= 255.0
            if img.ndimension() == 3: img = img.unsqueeze(0)

            with self.model_lock:
                pred = self.model(img, augment=False)[0]
                pred = non_max_suppression_kpt(pred, CONF_THRESHOLD, IOU_THRESHOLD, classes=0, kpt_label=True)

            det = pred[0]
            if len(det):
                scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=True)
                scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=True, step=3)
                
                det = det[:1] 
                for *xyxy, conf, cls in det[:, :6]:
                    kpts = det[0, 6:]
                    plot_one_box(xyxy, im0, label=f'ID{self.cam_id} P{conf:.2f}', color=colors(self.cam_id, True), 
                                 line_thickness=2, kpt_label=True, kpts=kpts, steps=3, orig_shape=im0.shape[:2])
                    
                    # 邏輯判斷區
                    hand_alert_signal = False 

                    # 事件錄影
                    if self.fps_initialized:
                        if self.preEvent_queue.full(): self.preEvent_queue.get()
                        self.preEvent_queue.put(im0)

                        if hand_alert_signal: 
                            self.is_event_recording = True
                            self.event_start_time = time.time()
                            self.event_filename = str(self.save_dir / f'Event_Cam{self.cam_id}_{datetime.now().strftime("%H%M%S")}.mp4')
                            
                            temp_list = list(self.preEvent_queue.queue)
                            for q_frame in temp_list:
                                video_saver.write_frame(self.event_filename, q_frame, self.rec_fps, w, h)

                        if self.is_event_recording:
                            video_saver.write_frame(self.event_filename, im0, self.rec_fps, w, h)
                            if time.time() - self.event_start_time > 10:
                                self.is_event_recording = False

            # FPS 計算
            elapsed_time = time.time() - self.prev_time
            self.sys_fps = 1 / elapsed_time if elapsed_time > 0 else 0
            
            if not self.fps_initialized:
                self.fps_records.append(self.sys_fps)
                if len(self.fps_records) == 50:
                    avg_fps = round(sum(self.fps_records) / len(self.fps_records))
                    self.rec_fps = max(avg_fps, 5)
                    self.preEvent_queue = queue.Queue(maxsize=int(self.rec_fps * 8))
                    self.fps_initialized = True
            else:
                video_saver.write_frame(self.video_procFile, im0, self.rec_fps, w, h)


# ============ [5] 主視窗 UI ============
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NX Edge AI Server - Surveillance Monitor")
        self.resize(1280, 720)
        self.setStyleSheet("background-color: #2b2b2b; color: #ffffff;")

        # --- UI 元件 ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # 1. 影像區 (左右各一)
        video_layout = QHBoxLayout()
        self.label_cam0 = self.create_video_label("Cam 0 (Waiting...)")
        self.label_cam1 = self.create_video_label("Cam 1 (Waiting...)")
        
        # 用 GroupBox 包起來比較美觀
        grp_cam0 = QGroupBox("Camera 0")
        grp_cam0.setLayout(QVBoxLayout())
        grp_cam0.layout().addWidget(self.label_cam0)
        
        grp_cam1 = QGroupBox("Camera 1")
        grp_cam1.setLayout(QVBoxLayout())
        grp_cam1.layout().addWidget(self.label_cam1)

        video_layout.addWidget(grp_cam0)
        video_layout.addWidget(grp_cam1)
        main_layout.addLayout(video_layout, stretch=4)

        # 2. 資訊日誌區
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas;")
        self.log_box.setMaximumHeight(150)
        main_layout.addWidget(self.log_box, stretch=1)

        # --- 初始化後端 ---
        self.init_backend()

    def create_video_label(self, text):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("background-color: #000000; border: 2px solid #555;")
        lbl.setMinimumSize(320, 240)
        lbl.setScaledContents(True) # 讓影像自動縮放填滿
        return lbl

    def init_backend(self):
        
        # 啟動 VideoSaver
        video_saver.start()

        # 啟動接收器
        self.receivers = []
        for config in RPI_CONFIGS:
            rx = StreamReceiver(config['ip'], config['port'], config['mapped_id'])
            rx.start()
            self.receivers.append(rx)

        # 載入模型 (在 Main Thread 載入，或另開 Loading Thread)
        self.log("Loading YOLOv7 Model...")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = attempt_load('thermalModel_v5b_100epoch.pt', map_location=self.device)
        if self.device.type != 'cpu': self.model.half()
        self.model.eval()
        self.model_lock = threading.Lock()
        
        self.save_dir = increment_path(Path("runs/detect"), exist_ok=False)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log(f"Model Loaded. Device: {self.device}")

        # 啟動動態偵測 Timer (取代原本的 While True)
        self.active_detectors = {}
        self.discovery_timer = QTimer()
        self.discovery_timer.timeout.connect(self.check_new_cameras)
        self.discovery_timer.start(500) # 每 0.5 秒檢查一次

    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.append(f"[{timestamp}] {msg}")
        # 自動捲動到底部
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def check_new_cameras(self):
        """ 檢查是否有新連線的相機 ID 出現，並啟動推論執行緒 """
        current_ids = []
        with frame_lock:
            current_ids = list(latest_frames.keys())

        for cid in current_ids:
            if cid not in self.active_detectors:
                self.log(f"Detected Active Stream ID: {cid}. Spawning Detector...")
                t = DetectThread(cid, self.model, self.model_lock, self.device, 
                                 self.save_dir, self.model.names, self.device.type != 'cpu', 640)
                t.start()
                self.active_detectors[cid] = t

    @pyqtSlot(int, np.ndarray, float)
    def update_video_ui(self, cam_id, frame, fps):
        """ 接收 Worker 傳來的影像並更新 UI """
        # 繪製 FPS (其實 DetectThread 已經畫在 frame 上了，這裡可以直接轉檔)
        # 如果需要在 UI 上額外顯示 FPS 標籤，可以在這裡處理
        
        # OpenCV (BGR) -> Qt (RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # 更新對應的 Label
        if cam_id == 0:
            self.label_cam0.setPixmap(QPixmap.fromImage(q_img))
        elif cam_id == 1:
            self.label_cam1.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        """ 視窗關閉時的清理工作 """
        self.log("Stopping System...")
        STOP_EVENT.set()
        video_saver.stop()
        event.accept()

# ============ [6] 程式進入點 ============
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # 設定字型 (可選)
    font = QFont("Arial", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())