import sys
import os
import cv2
import time
import shutil
import threading
import torch
import numpy as np
from datetime import date, datetime, timedelta  # 【新增】timedelta 用於計算午夜時間差
from pathlib import Path
import traceback
import json # 讀取 config.json

# PyQt6 UI 模組
from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QProgressBar, 
    QVBoxLayout, QHBoxLayout, QSplitter, QMessageBox, 
    QTreeView, QPushButton, QSizePolicy, QFileDialog,
    QTextEdit, QMenu, QAbstractItemView, QHeaderView, QSlider
)
from PyQt6.QtGui import QImage, QPixmap, QAction, QFileSystemModel, QColor, QTextCursor, QFont
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, QDir

# 引入後端模組
from camera_thread import CameraThread
from detect_thread import DetectThread
from models.experimental import attempt_load
from utils.torch_utils import select_device
# import globals as gl

# 【新增】匯入獨立封裝的蜂鳴器模組
from buzzer_controller import BuzzerController


def get_executable_dir():
    """
    取得程式執行的真實目錄。
    支援: 一般 Python 執行、PyInstaller 打包後的 exe/bin
    """
    if getattr(sys, 'frozen', False):
        # 如果是打包後的執行檔
        return os.path.dirname(sys.executable)
    else:
        # 如果是原始 Python script
        return os.path.dirname(os.path.abspath(__file__))
    
# ==========================================
# 日誌系統 (Log System)
# ==========================================
class SystemLogger:
    def __init__(self, log_folder_name="Log"):

        # 【修改】使用絕對路徑，確保 Log 資料夾生成在 exe 旁邊
        base_dir = get_executable_dir()
        self.log_dir = os.path.join(base_dir, log_folder_name)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # 啟動時檢查是否需要清理舊 Log
        self.cleanup_old_logs()
        self.log("系統啟動 (System Startup)")

    def get_log_file_path(self):
        """ 取得當天的 Log 檔案路徑 (每天一份) """
        today_str = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"{today_str}.txt")

    def log(self, message, is_error=False):
        """ 寫入訊息到 Log 檔 """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            level = "ERROR" if is_error else "INFO"
            log_content = f"[{timestamp}] [{level}] {message}\n"
            
            # 寫入檔案 (Append 模式)
            with open(self.get_log_file_path(), "a", encoding="utf-8") as f:
                f.write(log_content)
                
            # 如果是嚴重錯誤，也印在終端機
            if is_error:
                print(log_content)
        except Exception as e:
            print(f"寫入 Log 失敗: {e}")

    def cleanup_old_logs(self):
        """ 檢查 Log 檔案數量，若超過 365 天則全部刪除 """
        try:
            files = [os.path.join(self.log_dir, f) for f in os.listdir(self.log_dir) if f.endswith(".txt")]
            
            if len(files) > 365:
                print(f"[Log系統] Log 檔案數量 ({len(files)}) 超過 365，執行清理...")
                self.log(f"Log 檔案過多，執行全數清理。")
                
                for f in files:
                    try:
                        os.remove(f)
                    except Exception as e:
                        print(f"刪除 {f} 失敗: {e}")
                
                # 清理完後重新記錄一筆
                self.log("Log 檔案已重置 (Reset logs after 365 days limit).")
        except Exception as e:
            print(f"Log 清理檢查失敗: {e}")

# 初始化全域 Logger
sys_logger = SystemLogger()

# 讀取 config 參數
def load_app_config(config_path='config.json'):
    base_dir = get_executable_dir()
    full_path = os.path.join(base_dir, config_path)
    if not os.path.exists(full_path):
        print(f'找不到{full_path}將套用系統預測值。')
        return {}
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f'設定檔解析失敗: {e}')
        return {}

# 捕捉全域未處理異常 (Global Exception Hook)
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print("!!! 系統發生未預期崩潰 !!!", error_msg)
    
    # 寫入 Log
    sys_logger.log(f"系統崩潰 (Uncaught Exception):\n{error_msg}", is_error=True)
    
    # 呼叫原始的例外處理 (通常會結束程式)
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

# 將 Hook 掛載到系統
sys.excepthook = handle_exception

# ==========================================
# 後端核心
# ==========================================
class BackendSystem:
    def __init__(self, config_data):
        self.config = config_data # 儲存全域設定字典

        # 【修改】使用絕對路徑，確保 Project 資料夾生成在 exe 旁邊
        exe_dir = get_executable_dir()
        # self.BASE_VIDEO_DIR = os.path.join(exe_dir, "Project")
        # 改成硬碟**********************************************
        self.BASE_VIDEO_DIR = "/mnt/usb_ssd/Project"
        base_path = Path(self.BASE_VIDEO_DIR)
        base_path.mkdir(parents=True, exist_ok=True) # 若權限正確，這行將順利執行
        # *****************************************************
        # 【修改】初始化外部蜂鳴器類別，並將主程式的 Log 系統對接進去
        self.buzzer = BuzzerController(pin=32, logger_callback=sys_logger.log)
        sys_logger.log("獨立蜂鳴器模組已掛載")

        # 1. 初始化 YOLO
        print("正在載入 Thermal 模型...")
        try:
            self.device = select_device('0')
            self.model = attempt_load('thermalModel_v4_100epoch.pt', map_location=self.device)
            if self.device.type != 'cpu':
                self.model.half()
            self.model(torch.zeros(1, 3, 640, 640).to(self.device).type_as(next(self.model.parameters())))
            self.model_lock = threading.Lock()
            sys_logger.log("Thermal 模型載入成功")
        except Exception as e:
            sys_logger.log(f"Thermal 模型載入失敗: {e}", is_error=True)
            raise e
        
        # 2. 設定網路相機
        CAM0_PORT = 8000
        CAM1_PORT = 8001
        # print(f"啟動 ZMQ 接收: Port {CAM0_PORT}, {CAM1_PORT}")
        sys_logger.log(f"啟動相機接收 Port: {CAM0_PORT}, {CAM1_PORT}")

        self.cams = []
        self.cams.append(CameraThread(port=CAM0_PORT, name='Cam0', rotate=True))
        self.cams.append(CameraThread(port=CAM1_PORT, name='Cam1', rotate=True))

        # 3. 準備路徑
        base_path = Path(self.BASE_VIDEO_DIR)
        base_path.mkdir(parents=True, exist_ok=True)
        
        # 提供給 UI 初始化用的「今日路徑」
        self.current_ui_path = str((base_path / date.today().strftime("%Y/%m/%d")).resolve())
        print(f"系統根目錄: {self.BASE_VIDEO_DIR}")

        self.is_alarm_latched = self.config.get("system", {}).get("alarm_state_init", False)
        # 4. 啟動推論執行緒
        self.threads = []
        for i, cam in enumerate(self.cams):
            t = DetectThread(i, cam, self.model, self.model_lock, self.device,
                             base_path, cam.fps, self.model.names, 
                             self.device.type != 'cpu', imgsz=640,
                             config=self.config)
            t.start()
            self.threads.append(t)
    
    def get_frame(self, cam_id):
        if cam_id < len(self.threads):
            q = self.threads[cam_id].shared_frame_queue
            if not q.empty():
                return q.get() 
        return None

    def reset_alert(self):
        sys_logger.log("執行後端警報重置 (Reset Alert)")
        # gl.alarm_state = False
        self.is_alarm_latched = False
        for t in self.threads:
            if hasattr(t, 'reset_state'):
                t.reset_state()

    def stop(self):
        # print("--- 正在關閉系統 ---")
        sys_logger.log("系統正在關閉 (Backend Stopping)...")
        self.buzzer.cleanup()
        
        try:
            import detect_thread
            detect_thread.STOP_EVENT.set()
            
            import Jetson.GPIO as GPIO
            GPIO.cleanup()

        except:
            pass
            
        for cam in self.cams:
            if hasattr(cam, 'stop'): cam.stop()
        for t in self.threads:
            t.join(timeout=0.5)
        # print("--- 系統已安全中止 ---")
        sys_logger.log("系統已安全中止 (Shutdown Complete)")

# ==========================================
# 前端 UI
# ==========================================
class StandaloneWindow(QWidget):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.setWindowTitle("AIoT Surveillance System (Standalone)")
        self.resize(1700, 950)
        
        # 系統狀態變數
        self.ui_alert_active = False 
        self.current_date_str = datetime.now().strftime("%Y/%m/%d")
        
        # 【新增】蜂鳴器全域啟用狀態 (預設開啟)
        self.buzzer_enabled = True 

        self.setup_ui()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(30) 

        self.schedule_midnight_update()

    def setup_ui(self):
        # --- A. 相機顯示區 ---
        self.label_cam0 = QLabel("等待 Cam0 連線...")
        self.label_cam1 = QLabel("等待 Cam1 連線...")
        cam_style = """
            QLabel { background-color: black; color: white; font-size: 20px; border: 2px solid white; }
        """
        for lbl in [self.label_cam0, self.label_cam1]:
            lbl.setStyleSheet(cam_style)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFixedSize(640, 480) 
            lbl.setScaledContents(True)

        layout_cams = QHBoxLayout()
        layout_cams.addStretch() 
        layout_cams.addWidget(self.label_cam0)
        layout_cams.addWidget(self.label_cam1)
        layout_cams.addStretch() 
        layout_cams.setSpacing(20)

        # --- B. 控制區 ---
        self.label_clock = QLabel("00:00:00")
        self.label_clock.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont("Arial", 28, QFont.Weight.Bold)
        self.label_clock.setFont(font)
        self.label_clock.setStyleSheet("""
            QLabel { color: #2ecc71; background-color: #2c3e50; padding: 10px; border-radius: 10px; border: 2px solid #555; }
        """)

        # 主警報按鈕
        self.btn_alert = QPushButton("自傷行為\n(監控中)")
        self.btn_alert.setFixedSize(200, 160) # 稍微調小高度以容納靜音按鈕
        self.btn_alert.clicked.connect(self.reset_alert_state)
        self.btn_alert.setEnabled(False)
        
        self.style_normal = """
            QPushButton { background-color: #f0f0f0; color: #555555; font-size: 24px; font-weight: bold; border: 2px solid #999999; border-radius: 40px; }
        """
        self.style_alert = """
            QPushButton { background-color: #c0392b; color: white; font-size: 24px; font-weight: bold; border: 4px solid #e74c3c; border-radius: 40px; }
        """
        self.btn_alert.setStyleSheet(self.style_normal)

        # 【新增】蜂鳴器獨立開關按鈕
        self.btn_buzzer_toggle = QPushButton("🔊 蜂鳴器：已開啟\n(點擊靜音)")
        self.btn_buzzer_toggle.setFixedSize(200, 80)
        self.btn_buzzer_toggle.clicked.connect(self.toggle_buzzer)
        
        self.style_buzzer_on = """
            QPushButton { background-color: #27ae60; color: white; font-size: 16px; font-weight: bold; border-radius: 20px; border: 2px solid #2ecc71; }
            QPushButton:hover { background-color: #2ecc71; }
        """
        self.style_buzzer_off = """
            QPushButton { background-color: #7f8c8d; color: white; font-size: 16px; font-weight: bold; border-radius: 20px; border: 2px solid #95a5a6; }
            QPushButton:hover { background-color: #95a5a6; }
        """
        self.btn_buzzer_toggle.setStyleSheet(self.style_buzzer_on)

        # 【新增】音量標籤與拉霸條 (Volume Slider) ******************************************
        self.label_volume = QLabel("🔊 音量: 100%")
        self.label_volume.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        self.label_volume.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.slider_volume = QSlider(Qt.Orientation.Horizontal)
        self.slider_volume.setRange(0, 100) # UI 顯示 0% 到 100%
        self.slider_volume.setValue(100)    # 預設最大音量
        self.slider_volume.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_volume.setTickInterval(10)
        self.slider_volume.setStyleSheet("""
            QSlider::groove:horizontal { border: 1px solid #bbb; background: white; height: 10px; border-radius: 4px; }
            QSlider::sub-page:horizontal { background: #3498db; border: 1px solid #777; height: 10px; border-radius: 4px; }
            QSlider::handle:horizontal { background: #2980b9; border: 1px solid #777; width: 18px; margin-top: -5px; margin-bottom: -5px; border-radius: 9px; }
        """)
        self.slider_volume.valueChanged.connect(self.change_volume)
        # ************************************************************************************

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("""
            QTextEdit { background-color: white; font-size: 16px; border: 2px solid #1a5276; }
        """)
        self.log_area.setPlaceholderText("系統啟動... 等待事件觸發...")
        self.log_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # 左下角控制項佈局
        vbox_btn_clock = QVBoxLayout()
        vbox_btn_clock.addWidget(self.label_clock)
        vbox_btn_clock.addWidget(self.btn_alert)
        vbox_btn_clock.addSpacing(10)
        vbox_btn_clock.addWidget(self.btn_buzzer_toggle) # 加入靜音開關
        # 將音量控制加入佈局 ************************************
        vbox_btn_clock.addWidget(self.label_volume)
        vbox_btn_clock.addWidget(self.slider_volume)
        # ****************************************************
        vbox_btn_clock.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout_controls = QHBoxLayout()
        layout_controls.addLayout(vbox_btn_clock)
        layout_controls.addWidget(self.log_area)
        
        layout_left_panel = QVBoxLayout()
        layout_left_panel.addStretch(1) 
        layout_left_panel.addLayout(layout_cams)     
        layout_left_panel.addSpacing(10)
        layout_left_panel.addLayout(layout_controls, stretch=1) 

        container_left = QWidget()
        container_left.setLayout(layout_left_panel)

        main_layout = QHBoxLayout()
        # main_layout.addWidget(splitter)
        main_layout.addWidget(container_left)

        self.setLayout(main_layout)
        self.setStyleSheet("background-color: #2c3e50;")

    # ==========================================
    # 邏輯功能區
    # ==========================================
    # 【新增】跨日更新機制：每天午夜自動切換錄影目錄，並在 UI 上反映
    def schedule_midnight_update(self):
        now = datetime.now()
        tomorrow = now + timedelta(days=1)
        midnight = datetime(year=tomorrow.year, month=tomorrow.month, day=tomorrow.day, hour=0, minute=0, second=5)
        ms_until_midnight = int((midnight - now).total_seconds() * 1000)
        QTimer.singleShot(ms_until_midnight, self.perform_date_update)
    
    # 【修改】跨日更新後的動作：切換錄影目錄、更新 UI 顯示、記錄日誌
    def perform_date_update(self):
        now_str = datetime.now().strftime("%Y/%m/%d")
        if now_str != self.current_date_str:
            sys_logger.log(f"跨日更新: {self.current_date_str} -> {now_str}")
            self.current_date_str = now_str
            
            # self.file_label.setText(f"今日錄影目錄: {self.current_date_str}")
            new_path = os.path.join(self.backend.BASE_VIDEO_DIR, now_str)
            if not os.path.exists(new_path): os.makedirs(new_path)
            # self.file_tree.setRootIndex(self.file_model.index(new_path))
            self.log_area.append(f"[{datetime.now().strftime('%H:%M:%S')}] 跨日更新完成，切換至新目錄。")
            self.log_area.moveCursor(QTextCursor.MoveOperation.End)
        self.schedule_midnight_update()

    # 【修改】更新畫面同時檢查是否有警報訊息，並觸發 UI 反應
    def update_frames(self):
        current_time_str = datetime.now().strftime("%H:%M:%S")
        self.label_clock.setText(current_time_str)
        
        detected_alert_msg = None
        data0 = self.backend.get_frame(0)
        if data0:
            frame, alert_state = data0
            self.show_frame(self.label_cam0, frame)
            if alert_state: detected_alert_msg = alert_state

        data1 = self.backend.get_frame(1)
        if data1:
            frame, alert_state = data1
            self.show_frame(self.label_cam1, frame)
            if alert_state: detected_alert_msg = alert_state 
            
        if detected_alert_msg and not self.ui_alert_active:
            self.trigger_alert_ui(detected_alert_msg) 
    # 【修改】在 update_frames 中加入 alert_state 的檢查，確保 UI 反應與後端狀態同步
    def show_frame(self, label, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg))

    # 護理人員手動操作全域蜂鳴器開關
    def toggle_buzzer(self):
        """處理護理人員按下靜音/開啟蜂鳴器的邏輯"""
        self.buzzer_enabled = not self.buzzer_enabled
        
        if self.buzzer_enabled:
            # 狀態：開啟
            self.btn_buzzer_toggle.setText("🔊 蜂鳴器：已開啟\n(點擊靜音)")
            self.btn_buzzer_toggle.setStyleSheet(self.style_buzzer_on)
            sys_logger.log("護理人員操作：開啟蜂鳴器硬體功能")
            
            # 若目前正在警報中，恢復發聲
            if self.ui_alert_active:
                duration = self.backend.config.get('alert', {}).get('buzzer_duration_sec', 180)
                self.backend.buzzer.start_alarm()
        else:
            # 狀態：靜音
            self.btn_buzzer_toggle.setText("🔇 蜂鳴器：已靜音\n(點擊開啟)")
            self.btn_buzzer_toggle.setStyleSheet(self.style_buzzer_off)
            sys_logger.log("護理人員操作：靜音蜂鳴器硬體功能")
            
            # 強制停止目前可能正在發出的聲音
            self.backend.buzzer.stop_alarm()
    
    # 觸發警報時的動作
    def trigger_alert_ui(self, alert_type=''):
        self.ui_alert_active = True
        self.btn_alert.setText("⚠️ 異常發生 ⚠️\n點擊重置")
        self.btn_alert.setStyleSheet(self.style_alert)
        self.btn_alert.setEnabled(True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] 偵測到【{alert_type}】行為！"
        
        # 【修改】依據全域開關決定是否發出實體聲音
        if self.buzzer_enabled:
            # self.backend.buzzer.start_alarm()
            duration = self.backend.config.get('alert', {}).get('buzzer_duration_sec', 180)
            # 👇 【修復重點】補上這行，真正呼叫硬體並傳入 3 分鐘 (180秒) 參數
            self.backend.buzzer.start_alarm(duration_sec=duration)
            log_msg += f'蜂鳴器已觸發 (預計響鈴{duration}秒)'
        else:
            log_msg += " (蜂鳴器目前為靜音狀態)"
            
        self.log_area.append(log_msg)
        self.log_area.moveCursor(QTextCursor.MoveOperation.End)
        sys_logger.log(f"偵測到行為: {alert_type} (發聲狀態: {self.buzzer_enabled})")

    # 解除警報時的動作
    def reset_alert_state(self):
        sys_logger.log("使用者手動重置警報")

        # 1. 呼叫後端重置推論狀態
        self.backend.reset_alert()
        # 2. 強制停止硬體蜂鳴器 (綁定消音邏輯)
        self.backend.buzzer.stop_alarm()

        self.ui_alert_active = False
        self.btn_alert.setText("自傷行為\n(監控中)")
        self.btn_alert.setStyleSheet(self.style_normal)
        self.btn_alert.setEnabled(False)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_area.append(f"[{timestamp}] 警報已手動解除。")
    
    # 【新增】處理拉霸條數值變更 **********************************
    def change_volume(self, value):
        self.label_volume.setText(f"🔊 音量: {value}%")
        
        # 硬體防呆：將 UI 的 0~100 映射為 PWM 的 25~50 Duty Cycle
        # 100 / 4 = 25 的增量，基底為 25（最小安全音量） 
        mapped_duty_cycle = int(value / 4 + 25) 
        
        # 呼叫後端模組進行實體更新
        if hasattr(self.backend.buzzer, 'set_volume'):
            self.backend.buzzer.set_volume(mapped_duty_cycle)

   
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
    # 使用 try-except 包裹整個主程式，作為最後一道防線
    try:
        sys_config = load_app_config()
        app = QApplication(sys.argv)
        backend = BackendSystem(config_data=sys_config)
        window = StandaloneWindow(backend)
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        # 這裡會捕捉到 Backend 初始化失敗等嚴重錯誤
        error_msg = traceback.format_exc()
        print(f"主程式啟動失敗: {error_msg}")
        if 'sys_logger' in globals():
            sys_logger.log(f"主程式啟動失敗:\n{error_msg}", is_error=True)