import sys
import os
import cv2
import time
import threading
import torch
import numpy as np
from datetime import date, datetime
from pathlib import Path

# PyQt6 UI 模組
from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QProgressBar, 
    QVBoxLayout, QHBoxLayout, QSplitter, QMessageBox, 
    QTreeWidget, QTreeWidgetItem, QPushButton, QSizePolicy, QFileDialog
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, QThread, pyqtSignal

# 引入您原本的後端模組
from camera_thread import CameraThread
from detect_thread import DetectThread
from models.experimental import attempt_load
from utils.torch_utils import select_device
import globals as gl

# ==========================================
# 後端核心 (原本的 MainThread，現在被整合進來)
# ==========================================
class BackendSystem:
    def __init__(self):
        self.BASE_VIDEO_DIR = "/home/aiot/Project"
        
        # 1. 初始化 YOLO
        print("正在載入 YOLO 模型...")
        self.device = select_device('0')
        self.model = attempt_load('thermalModel_v4_100epoch.pt', map_location=self.device)
        if self.device.type != 'cpu':
            self.model.half()
        # 熱身
        self.model(torch.zeros(1, 3, 640, 640).to(self.device).type_as(next(self.model.parameters())))
        self.model_lock = threading.Lock()

        # 2. 設定網路相機 (ZMQ 接收 RJ45 訊號)
        CAM0_PORT = 5555
        CAM1_PORT = 5556
        print(f"啟動 ZMQ 接收: Port {CAM0_PORT}, {CAM1_PORT}")
        
        self.cams = []
        self.cams.append(CameraThread(port=CAM0_PORT, name='Cam0', rotate=True))
        self.cams.append(CameraThread(port=CAM1_PORT, name='Cam1', rotate=True))

        # 3. 準備錄影路徑
        today = date.today().strftime("%Y/%m/%d")
        base_path = Path(self.BASE_VIDEO_DIR)
        save_dir = base_path / today
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"錄影路徑: {save_dir}")

        # 4. 啟動推論執行緒
        self.threads = []
        for i, cam in enumerate(self.cams):
            t = DetectThread(i, cam, self.model, self.model_lock, self.device,
                             save_dir, cam.fps, self.model.names, 
                             self.device.type != 'cpu', imgsz=640)
            t.start()
            self.threads.append(t)

    def get_frame(self, cam_id):
        """ UI 透過這個方法直接拿影像，不用經過 WebSocket """
        if cam_id < len(self.threads):
            # shared_frame_queue 裡面放的是 (frame, alert_status)
            q = self.threads[cam_id].shared_frame_queue
            if not q.empty():
                return q.get() # 取出最新的
        return None

    def reset_alert(self):
        gl.alarm_state = False
        for t in self.threads:
            t.reset_state()

    # 加在 BackendSystem 類別裡面的最後面
    def stop(self):
        print("--- 正在關閉系統 ---")
        
        # 1. 告訴 DetectThread (AI) 停止運作
        # 需確保已從 detect_thread 匯入 STOP_EVENT
        import detect_thread
        detect_thread.STOP_EVENT.set()
 
        # 2. 停止 CameraThread (ZMQ 接收)
        for cam in self.cams:
            if hasattr(cam, 'stop'):
                cam.stop()

        # 3. 等待所有執行緒結束 (避免記憶體洩漏)
        for t in self.threads:
            t.join(timeout=0.5)
            
        print("--- 系統已安全中止 ---")

# ==========================================
# 前端 UI (原本的 MainWindow)
# ==========================================
class StandaloneWindow(QWidget):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend  # 持有後端物件的參照
        
        self.setWindowTitle("AIoT Surveillance System (Standalone)")
        self.resize(1500, 900)
        self.setup_ui()
        
        # --- 關鍵修改：使用 QTimer 定時更新畫面 ---
        # 每 30ms (約 33 FPS) 去後端拿一次影像
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(30) 

        # 每 1 秒更新檔案列表
        self.file_timer = QTimer(self)
        self.file_timer.timeout.connect(self.load_local_files)
        self.file_timer.start(1000)

    def setup_ui(self):
        # ... (這裡與您原本的 UI Layout 程式碼幾乎一樣，省略重複部分以節省篇幅) ...
        # 建立 Cam0, Cam1 Label
        self.label_cam0 = QLabel("Waiting for Cam0...")
        self.label_cam1 = QLabel("Waiting for Cam1...")
        for lbl in [self.label_cam0, self.label_cam1]:
            lbl.setStyleSheet("background: black; color: white; qproperty-alignment: AlignCenter;")
            lbl.setMinimumHeight(300)
            lbl.setScaledContents(True)

        hbox_cams = QHBoxLayout()
        hbox_cams.addWidget(self.label_cam0)
        hbox_cams.addWidget(self.label_cam1)
        
        # 警報按鈕
        self.btn_alert = QPushButton("安全\n無警報")
        self.btn_alert.setFixedWidth(200)
        self.btn_alert.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.btn_alert.clicked.connect(self.reset_alert_state)
        self.normal_style = "background-color: #e0e0e0; color: #555555; font-size: 24px; border-radius: 15px;"
        self.alert_style = "background-color: #ff3333; color: white; font-size: 28px; border-radius: 15px;"
        self.btn_alert.setStyleSheet(self.normal_style)

        # 檔案列表
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["錄影日期 / 檔案名稱"])
        self.tree.itemDoubleClicked.connect(self.open_local_file) # 改成直接開啟檔案
        
        # Layout 組合
        hbox_bottom = QHBoxLayout()
        hbox_bottom.addWidget(self.btn_alert)
        hbox_bottom.addWidget(self.tree)
        
        splitter = QSplitter(Qt.Orientation.Vertical)
        cams_widget = QWidget(); cams_widget.setLayout(hbox_cams)
        bottom_widget = QWidget(); bottom_widget.setLayout(hbox_bottom)
        splitter.addWidget(cams_widget)
        splitter.addWidget(bottom_widget)
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def update_frames(self):
        """ 核心迴圈：從 Backend 拿資料更新 UI """
        # 更新 Cam 0
        data0 = self.backend.get_frame(0)
        if data0:
            frame, is_alert = data0
            self.show_frame(self.label_cam0, frame)
            if is_alert: self.trigger_ui_alert()

        # 更新 Cam 1
        data1 = self.backend.get_frame(1)
        if data1:
            frame, is_alert = data1
            self.show_frame(self.label_cam1, frame)
            if is_alert: self.trigger_ui_alert()

    def show_frame(self, label, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg))

    def trigger_ui_alert(self):
        if "異常" not in self.btn_alert.text():
            self.btn_alert.setText("⚠️ 異常 ⚠️\n點擊清除")
            self.btn_alert.setStyleSheet(self.alert_style)

    def reset_alert_state(self):
        # 直接呼叫後端函式，不用發 HTTP Request
        self.backend.reset_alert()
        self.btn_alert.setText("安全\n無警報")
        self.btn_alert.setStyleSheet(self.normal_style)

    def load_local_files(self):
        """ 直接讀取本機資料夾，取代 Flask API """
        video_dir = self.backend.BASE_VIDEO_DIR
        # 這裡可以使用 os.walk 邏輯來更新 TreeWidget
        # (邏輯與之前類似，但直接操作 self.tree，不贅述)
        pass

    def open_local_file(self, item, col):
        """ 直接用系統播放器打開檔案，不需要下載 """
        # 假設您有將完整路徑存入 item.data
        # path = item.data(0, Qt.ItemDataRole.UserRole)
        # if os.path.exists(path):
        #     os.system(f"xdg-open {path}") # Linux 開檔指令
        pass

    # 加在 StandaloneWindow 類別裡面的最後面
    def closeEvent(self, event):
        """
        當使用者按下視窗右上角的 X 時，會自動觸發此函式
        """
        # (選項 A) 直接關閉，不囉嗦
        # self.cleanup_and_close()
        # event.accept()

        # (選項 B) 跳出確認視窗，防止誤觸 (建議使用)
        reply = QMessageBox.question(
            self, 
            '關閉確認', 
            "確定要結束監控系統嗎？\n(這將停止所有錄影與 AI 偵測)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.cleanup_and_close()
            event.accept()  # 接受關閉請求，視窗會消失
        else:
            event.ignore()  # 忽略關閉請求，視窗繼續運作

    def cleanup_and_close(self):
        """ 執行清理動作 """
        print("UI: 收到關閉訊號...")
        
        # 1. 停止 UI 計時器 (重要！否則 UI 消失後計時器還在跑會報錯)
        if self.timer.isActive():
            self.timer.stop()
        if self.file_timer.isActive():
            self.file_timer.stop()

        # 2. 呼叫後端停止
        self.backend.stop()

# ==========================================
# 主程式入口
# ==========================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 1. 先啟動後端系統
    backend = BackendSystem()
    
    # 2. 啟動 UI，並把後端交給它控制
    window = StandaloneWindow(backend)
    window.show()
    
    sys.exit(app.exec())