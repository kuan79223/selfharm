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

# PyQt6 UI 模組
from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QProgressBar, 
    QVBoxLayout, QHBoxLayout, QSplitter, QMessageBox, 
    QTreeView, QPushButton, QSizePolicy, QFileDialog,
    QTextEdit, QMenu, QAbstractItemView, QHeaderView
)
from PyQt6.QtGui import QImage, QPixmap, QAction, QFileSystemModel, QColor, QTextCursor, QFont
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, QDir

# 引入後端模組
from camera_thread import CameraThread
from detect_thread import DetectThread
from models.experimental import attempt_load
from utils.torch_utils import select_device
import globals as gl


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
    def __init__(self):
        # 【修改】使用絕對路徑，確保 Project 資料夾生成在 exe 旁邊
        exe_dir = get_executable_dir()
        self.BASE_VIDEO_DIR = os.path.join(exe_dir, "Project")
        
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

        # 4. 啟動推論執行緒
        self.threads = []
        for i, cam in enumerate(self.cams):
            t = DetectThread(i, cam, self.model, self.model_lock, self.device,
                             base_path, cam.fps, self.model.names, 
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
        sys_logger.log("執行後端警報重置 (Reset Alert)")
        gl.alarm_state = False
        for t in self.threads:
            if hasattr(t, 'reset_state'):
                t.reset_state()

    def stop(self):
        # print("--- 正在關閉系統 ---")
        sys_logger.log("系統正在關閉 (Backend Stopping)...")
        try:
            import detect_thread
            detect_thread.STOP_EVENT.set()
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
        
        self.ui_alert_active = False 
        self.current_date_str = datetime.now().strftime("%Y/%m/%d")

        self.setup_ui()
        
        # 定時更新影像 (30ms)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(30) 

        # 【修改 1】 啟動午夜更新排程 (不再使用每 10 秒的 Timer)
        self.schedule_midnight_update()

    def setup_ui(self):
        # --- A. 相機顯示區 ---
        self.label_cam0 = QLabel("等待 Cam0 連線...")
        self.label_cam1 = QLabel("等待 Cam1 連線...")
        
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
            lbl.setFixedSize(640, 480) 
            lbl.setScaledContents(True)

        layout_cams = QHBoxLayout()
        layout_cams.addStretch() 
        layout_cams.addWidget(self.label_cam0)
        layout_cams.addWidget(self.label_cam1)
        layout_cams.addStretch() 
        layout_cams.setSpacing(20)

        # --- A. 左下角元件：時鐘 (QLabel) ---
        self.label_clock = QLabel("00:00:00")
        self.label_clock.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 設定字型：Arial, 大小 28, 粗體
        font = QFont("Arial", 28, QFont.Weight.Bold)
        self.label_clock.setFont(font)
        # 設定外觀：綠字、深灰底、圓角邊框
        self.label_clock.setStyleSheet("""
            QLabel {
                color: #2ecc71; 
                background-color: #2c3e50;
                padding: 10px;
                border-radius: 10px;
                border: 2px solid #555;
            }
        """)

        # --- B. 控制區 ---
        self.btn_alert = QPushButton("自傷行為\n(監控中)")
        self.btn_alert.setFixedSize(200, 200)
        self.btn_alert.clicked.connect(self.reset_alert_state)
        
        # 【修改 2】 預設將按鈕設為不可點擊 (Disabled)
        self.btn_alert.setEnabled(False)
        
        # 定義樣式 (Normal 時顏色可以稍微淡一點表示未啟用，或保持白色但不可按)
        self.style_normal = """
            QPushButton {
                background-color: #f0f0f0; /* 稍微灰一點表示待機 */
                color: #555555;
                font-size: 24px;
                font-weight: bold;
                border: 2px solid #999999;
                border-radius: 100px;
            }
        """
        self.style_alert = """
            QPushButton {
                background-color: #c0392b;
                color: white;
                font-size: 24px;
                font-weight: bold;
                border: 4px solid #e74c3c;
                border-radius: 100px;
            }
        """
        self.btn_alert.setStyleSheet(self.style_normal)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("""
            QTextEdit {
                background-color: white;
                font-size: 16px;
                border: 2px solid #1a5276;
            }
        """)
        self.log_area.setPlaceholderText("系統啟動... 等待事件觸發...")
        self.log_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # 建立一個垂直 Layout 放按鈕跟時鐘
        vbox_btn_clock = QVBoxLayout()
        vbox_btn_clock.addWidget(self.label_clock) # 時鐘在上面
        vbox_btn_clock.addWidget(self.btn_alert)   # 按鈕在下面
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

        # -------------------------------------------------
        # 2. 右側區域
        # -------------------------------------------------
        target_path = self.backend.current_ui_path
        
        self.file_label = QLabel(f"今日錄影目錄: {self.current_date_str}")
        self.file_label.setStyleSheet("color: white; font-weight: bold; font-size: 14px; background: #154360; padding: 5px;")
        
        self.file_model = QFileSystemModel()
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        self.file_model.setRootPath(target_path)
        self.file_model.setFilter(QDir.Filter.NoDotAndDotDot | QDir.Filter.AllDirs | QDir.Filter.Files)
        self.file_model.setNameFilters(["*.mp4", "*.avi"]) 
        self.file_model.setNameFilterDisables(False)

        self.file_tree = QTreeView()
        self.file_tree.setModel(self.file_model)
        self.file_tree.setRootIndex(self.file_model.index(target_path))
        
        self.file_tree.setAlternatingRowColors(True)
        self.file_tree.setAnimated(True)
        self.file_tree.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        
        self.file_tree.setColumnHidden(2, True) 
        self.file_tree.setColumnHidden(3, True)
        self.file_tree.setColumnWidth(0, 280) 
        
        self.file_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_tree.customContextMenuRequested.connect(self.open_context_menu)
        self.file_tree.doubleClicked.connect(self.open_file_system)

        layout_right_panel = QVBoxLayout()
        layout_right_panel.addWidget(self.file_label)
        layout_right_panel.addWidget(self.file_tree)
        
        container_right = QWidget()
        container_right.setStyleSheet("background-color: #21618c;") 
        container_right.setLayout(layout_right_panel)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(container_left)
        splitter.addWidget(container_right)
        splitter.setSizes([1400, 300]) 

        main_layout = QHBoxLayout()
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        
        self.setStyleSheet("background-color: #2c3e50;")

    # ==========================================
    # 邏輯功能區
    # ==========================================

    def schedule_midnight_update(self):
        """ 【新增】 計算距離明天 00:00:05 還有多久，設定一次性定時器 """
        now = datetime.now()
        # 取得明天日期
        tomorrow = now + timedelta(days=1)
        # 設定為明天的 00:00:05 (多5秒緩衝，確保系統時間已跳轉)
        midnight = datetime(year=tomorrow.year, month=tomorrow.month, day=tomorrow.day, hour=0, minute=0, second=5)
        
        # 計算時間差 (秒)
        seconds_until_midnight = (midnight - now).total_seconds()
        ms_until_midnight = int(seconds_until_midnight * 1000)
        
        print(f"[系統排程] 將於 {seconds_until_midnight:.1f} 秒後 ({midnight}) 執行日期更新檢查")
        
        # 設定 SingleShot Timer (只執行一次)
        QTimer.singleShot(ms_until_midnight, self.perform_date_update)

    def perform_date_update(self):
        """ 執行跨日更新邏輯，並排程下一次檢查 """
        now_str = datetime.now().strftime("%Y/%m/%d")
        
        if now_str != self.current_date_str:
            # print(f"[UI] 執行跨日更新！ {self.current_date_str} -> {now_str}")
            sys_logger.log(f"跨日更新: {self.current_date_str} -> {now_str}")
            self.current_date_str = now_str
            
            # 更新介面
            self.file_label.setText(f"今日錄影目錄: {self.current_date_str}")
            new_path = os.path.join(self.backend.BASE_VIDEO_DIR, now_str)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            self.file_tree.setRootIndex(self.file_model.index(new_path))
            
            self.log_area.append(f"[{datetime.now().strftime('%H:%M:%S')}] 跨日更新完成，切換至新目錄。")
            self.log_area.moveCursor(QTextCursor.MoveOperation.End)
        
        # 遞迴呼叫，排程「後天」的午夜更新 (形成每日循環)
        self.schedule_midnight_update()

    def update_frames(self):

        # 1. 更新時間
        current_time_str = datetime.now().strftime("%H:%M:%S")
        self.label_clock.setText(current_time_str)
        
        # 【修改】用變數來接收警報類型 (字串)
        detected_alert_msg = None
        data0 = self.backend.get_frame(0)
        if data0:
            frame, alert_state = data0 # 這裡的 alert_state 變成了字串 (如 "手部") 或 None
            self.show_frame(self.label_cam0, frame)
            if alert_state:
                detected_alert_msg = alert_state # 記錄警報類型

        data1 = self.backend.get_frame(1)
        if data1:
            frame, alert_state = data1
            self.show_frame(self.label_cam1, frame)
            if alert_state: 
                # 如果兩機同時觸發，可以用字串串接，或直接覆蓋 (這裡示範覆蓋)
                detected_alert_msg = alert_state
            
        if detected_alert_msg and not self.ui_alert_active:
            self.trigger_alert_ui(detected_alert_msg) # <--- 傳入類型字串

        
    def show_frame(self, label, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg))

    def trigger_alert_ui(self, alert_type=''):
        self.ui_alert_active = True
        self.btn_alert.setText("⚠️ 異常發生 ⚠️\n點擊重置")
        self.btn_alert.setStyleSheet(self.style_alert)
        # 【修改 3】 警報發生時，啟用按鈕讓使用者可以點擊
        self.btn_alert.setEnabled(True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 【修改】Log 訊息顯示具體類型
        log_msg = f"[{timestamp}] 偵測到【{alert_type}】自傷行為！"
        self.log_area.append(log_msg)
        self.log_area.moveCursor(QTextCursor.MoveOperation.End)
        # 寫入系統 Log
        sys_logger.log("偵測到自傷行為！")

    def reset_alert_state(self):
        # print("使用者手動重置警報...")
        sys_logger.log("使用者手動重置警報")
        self.backend.reset_alert()
        self.ui_alert_active = False
        self.btn_alert.setText("自傷行為\n(監控中)")
        self.btn_alert.setStyleSheet(self.style_normal)
        
        # 【修改 4】 重置後，再次禁用按鈕
        self.btn_alert.setEnabled(False)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_area.append(f"[{timestamp}] 警報已手動解除。")
        
    def open_context_menu(self, position):
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
        file_path = self.file_model.filePath(index)
        import subprocess
        try:
            print(f"嘗試使用 VLC 播放: {file_path}")
            subprocess.Popen(['vlc', file_path])
        except Exception as e:
            QMessageBox.warning(self, "錯誤", f"無法開啟 VLC: {e}")

    def export_file(self, source_path):
        is_dir = os.path.isdir(source_path)
        if is_dir:
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
            file_name = os.path.basename(source_path)
            target_file, _ = QFileDialog.getSaveFileName(self, "另存新檔", file_name)
            if target_file:
                try:
                    shutil.copy2(source_path, target_file)
                    QMessageBox.information(self, "完成", f"檔案已匯出至:\n{target_file}")
                except Exception as e:
                    QMessageBox.warning(self, "錯誤", f"匯出失敗: {e}")

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
        app = QApplication(sys.argv)
        backend = BackendSystem()
        window = StandaloneWindow(backend)
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        # 這裡會捕捉到 Backend 初始化失敗等嚴重錯誤
        error_msg = traceback.format_exc()
        print(f"主程式啟動失敗: {error_msg}")
        if 'sys_logger' in globals():
            sys_logger.log(f"主程式啟動失敗:\n{error_msg}", is_error=True)