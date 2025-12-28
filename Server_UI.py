import sys
import os
import cv2
import time
import shutil  # 用於檔案複製 (匯出功能)
import threading
import torch
import numpy as np
from datetime import date, datetime
from pathlib import Path

# PyQt6 UI 模組
from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QProgressBar, 
    QVBoxLayout, QHBoxLayout, QSplitter, QMessageBox, 
    QTreeView, QPushButton, QSizePolicy, QFileDialog,
    QTextEdit, QMenu, QAbstractItemView, QHeaderView
)
from PyQt6.QtGui import QImage, QPixmap, QAction, QFileSystemModel, QColor, QTextCursor
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, QDir

# 引入後端模組
from camera_thread import CameraThread
from detect_thread import DetectThread
from models.experimental import attempt_load
from utils.torch_utils import select_device
import globals as gl

# ==========================================
# 後端核心 (負責 AI 與 影像採集)
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
        self.model(torch.zeros(1, 3, 640, 640).to(self.device).type_as(next(self.model.parameters())))
        self.model_lock = threading.Lock()

        # 2. 設定網路相機 (ZMQ)
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
        if cam_id < len(self.threads):
            q = self.threads[cam_id].shared_frame_queue
            if not q.empty():
                return q.get() 
        return None

    def reset_alert(self):
        gl.alarm_state = False
        for t in self.threads:
            if hasattr(t, 'reset_state'):
                t.reset_state()

    def stop(self):
        print("--- 正在關閉系統 ---")
        import detect_thread
        detect_thread.STOP_EVENT.set()
        for cam in self.cams:
            if hasattr(cam, 'stop'): cam.stop()
        for t in self.threads:
            t.join(timeout=0.5)
        print("--- 系統已安全中止 ---")

# ==========================================
# 前端 UI (依照新設計圖實作)
# ==========================================
class StandaloneWindow(QWidget):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        
        self.setWindowTitle("AIoT Surveillance System (Standalone)")
        # 設定視窗預設最大化，或是固定解析度
        self.resize(1600, 900)
        
        # 用來記錄 UI 目前是否處於紅燈狀態 (避免重複刷新 Log)
        self.ui_alert_active = False 

        self.setup_ui()
        
        # 定時更新影像 (30ms)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(30) 

    def setup_ui(self):
        # -------------------------------------------------
        # 1. 左側區域：上(相機) + 下(按鈕與日誌)
        # -------------------------------------------------
        
        # --- A. 相機顯示區 (Top Left) ---
        self.label_cam0 = QLabel("等待 Cam0 連線...")
        self.label_cam1 = QLabel("等待 Cam1 連線...")
        
        # 設定樣式：深藍色背景 (參考圖片)
        cam_style = """
            QLabel {
                background-color: black; 
                color: white; 
                font-size: 20px;
                border: 2px solid white;
            }
        """
        for lbl in [self.label_cam0, self.label_cam1]:
            lbl.setStyleSheet(cam_style)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setMinimumSize(400, 300)
            lbl.setScaledContents(True)

        layout_cams = QHBoxLayout()
        layout_cams.addWidget(self.label_cam0)
        layout_cams.addWidget(self.label_cam1)
        layout_cams.setSpacing(5)

        # --- B. 控制區 (Bottom Left) ---
        
        # 自傷行為按鈕 (圓形/方形切換)
        self.btn_alert = QPushButton("自傷行為\n(監控中)")
        self.btn_alert.setFixedSize(200, 200) # 設定為正方形，可用 border-radius 變圓
        self.btn_alert.clicked.connect(self.reset_alert_state)
        
        # 定義樣式 (Normal: 白底; Alert: 紅底)
        self.style_normal = """
            QPushButton {
                background-color: white;
                color: black;
                font-size: 24px;
                font-weight: bold;
                border: 2px solid black;
                border-radius: 100px; /* 變圓形 */
            }
        """
        self.style_alert = """
            QPushButton {
                background-color: #c0392b; /* 紅色 */
                color: white;
                font-size: 24px;
                font-weight: bold;
                border: 4px solid #e74c3c;
                border-radius: 100px; /* 變圓形 */
            }
        """
        self.btn_alert.setStyleSheet(self.style_normal)

        # 滾動日誌視窗 (QTextEdit)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True) # 唯讀
        self.log_area.setStyleSheet("""
            QTextEdit {
                background-color: white;
                font-size: 16px;
                border: 2px solid #1a5276;
            }
        """)
        self.log_area.setPlaceholderText("系統啟動... 等待事件觸發...")
        self.log_area.setMinimumHeight(200)

        layout_controls = QHBoxLayout()
        layout_controls.addWidget(self.btn_alert)
        layout_controls.addWidget(self.log_area)
        
        # 左側總成 (Vertical)
        layout_left_panel = QVBoxLayout()
        layout_left_panel.addLayout(layout_cams, stretch=3)     # 上半部佔 3 等份
        layout_left_panel.addLayout(layout_controls, stretch=1) # 下半部佔 1 等份

        container_left = QWidget()
        container_left.setLayout(layout_left_panel)

        # -------------------------------------------------
        # 2. 右側區域：檔案總管 (NX端資料夾)
        # -------------------------------------------------
        self.file_label = QLabel(f"NX端資料夾: {self.backend.BASE_VIDEO_DIR}")
        self.file_label.setStyleSheet("color: white; font-weight: bold; font-size: 14px; background: #154360; padding: 5px;")
        
        # 使用 QFileSystemModel 來顯示真實檔案系統
        self.file_model = QFileSystemModel()
        root_path = self.backend.BASE_VIDEO_DIR
        
        # 如果路徑不存在，先建立以免報錯
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        self.file_model.setRootPath(root_path)
        # 設定只顯示 MP4 檔案與資料夾
        self.file_model.setNameFilters(["*.mp4"])
        self.file_model.setNameFilterDisables(False)

        self.file_tree = QTreeView()
        self.file_tree.setModel(self.file_model)
        self.file_tree.setRootIndex(self.file_model.index(root_path))
        
        # 外觀設定
        self.file_tree.setAlternatingRowColors(True)
        self.file_tree.setAnimated(True)
        self.file_tree.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.file_tree.setColumnWidth(0, 300) # 檔名欄位寬一點
        
        # 啟用右鍵選單 (匯出功能)
        self.file_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_tree.customContextMenuRequested.connect(self.open_context_menu)
        
        # 雙擊開啟
        self.file_tree.doubleClicked.connect(self.open_file_system)

        layout_right_panel = QVBoxLayout()
        layout_right_panel.addWidget(self.file_label)
        layout_right_panel.addWidget(self.file_tree)
        
        container_right = QWidget()
        container_right.setStyleSheet("background-color: #21618c;") # 設定右側背景色
        container_right.setLayout(layout_right_panel)

        # -------------------------------------------------
        # 3. 主佈局：左右分割
        # -------------------------------------------------
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(container_left)
        splitter.addWidget(container_right)
        splitter.setStretchFactor(0, 3) # 左邊寬一點
        splitter.setStretchFactor(1, 1) # 右邊窄一點

        main_layout = QHBoxLayout()
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        
        # 設定整體背景色 (像圖片一樣的深藍底)
        self.setStyleSheet("background-color: #2c3e50;")

    # ==========================================
    # 邏輯功能區
    # ==========================================
    
    def update_frames(self):
        """ 定時從後端獲取影像與狀態 """
        alert_triggered = False
        
        # 更新 Cam 0
        data0 = self.backend.get_frame(0)
        if data0:
            frame, is_alert = data0
            self.show_frame(self.label_cam0, frame)
            if is_alert: alert_triggered = True

        # 更新 Cam 1
        data1 = self.backend.get_frame(1)
        if data1:
            frame, is_alert = data1
            self.show_frame(self.label_cam1, frame)
            if is_alert: alert_triggered = True
            
        # 處理警報邏輯 (Latching)
        if alert_triggered and not self.ui_alert_active:
            self.trigger_alert_ui()

    def show_frame(self, label, frame):
        # OpenCV (BGR) -> Qt (RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg))

    def trigger_alert_ui(self):
        """ 觸發紅燈與日誌紀錄 """
        self.ui_alert_active = True
        self.btn_alert.setText("⚠️ 異常發生 ⚠️\n點擊重置")
        self.btn_alert.setStyleSheet(self.style_alert)
        
        # 寫入日誌
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] 偵測到自傷行為！"
        self.log_area.append(log_msg)
        
        # 自動捲動到底部
        self.log_area.moveCursor(QTextCursor.MoveOperation.End)

    def reset_alert_state(self):
        """ 使用者按下按鈕重置 """
        print("使用者手動重置警報...")
        self.backend.reset_alert()
        self.ui_alert_active = False
        self.btn_alert.setText("自傷行為\n(監控中)")
        self.btn_alert.setStyleSheet(self.style_normal)
        
        # 寫入日誌
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_area.append(f"[{timestamp}] 警報已手動解除。")

    # --- 檔案總管功能 ---

    def open_context_menu(self, position):
        """ 右鍵選單：匯出檔案 """
        indexes = self.file_tree.selectedIndexes()
        if len(indexes) > 0:
            index = indexes[0]
            file_path = self.file_model.filePath(index)
            
            menu = QMenu()
            action_open = QAction("開啟 (Open)", self)
            action_export = QAction("匯出/下載 (Export)", self)
            
            action_open.triggered.connect(lambda: self.open_file_system(index))
            action_export.triggered.connect(lambda: self.export_file(file_path))
            
            menu.addAction(action_open)
            menu.addAction(action_export)
            menu.exec(self.file_tree.viewport().mapToGlobal(position))

    def open_file_system(self, index):
        """ 雙擊開啟檔案 """
        file_path = self.file_model.filePath(index)
        # 使用 Linux 預設程式開啟 (xdg-open)
        import subprocess
        try:
            subprocess.Popen(['xdg-open', file_path])
        except Exception as e:
            QMessageBox.warning(self, "錯誤", f"無法開啟檔案: {e}")

    def export_file(self, source_path):
        """ 模擬下載：將檔案複製到使用者選擇的地方 (如隨身碟) """
        is_dir = os.path.isdir(source_path)
        
        if is_dir:
            # 選擇匯出資料夾的位置
            target_dir = QFileDialog.getExistingDirectory(self, "選擇匯出目的地")
            if target_dir:
                folder_name = os.path.basename(source_path)
                full_target = os.path.join(target_dir, folder_name)
                try:
                    shutil.copytree(source_path, full_target)
                    QMessageBox.information(self, "完成", f"資料夾已匯出至:\n{full_target}")
                except Exception as e:
                    QMessageBox.warning(self, "錯誤", f"匯出失敗: {e}")
        else:
            # 選擇存檔檔名
            file_name = os.path.basename(source_path)
            target_file, _ = QFileDialog.getSaveFileName(self, "另存新檔", file_name)
            if target_file:
                try:
                    shutil.copy2(source_path, target_file)
                    QMessageBox.information(self, "完成", f"檔案已匯出至:\n{target_file}")
                except Exception as e:
                    QMessageBox.warning(self, "錯誤", f"匯出失敗: {e}")

    # --- 關閉程式 ---
    def closeEvent(self, event):
        reply = QMessageBox.question(self, '關閉確認', "確定要結束監控系統嗎？",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.cleanup_and_close()
            event.accept()
        else:
            event.ignore()

    def cleanup_and_close(self):
        if self.timer.isActive(): self.timer.stop()
        self.backend.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    backend = BackendSystem()
    window = StandaloneWindow(backend)
    window.show()
    sys.exit(app.exec())