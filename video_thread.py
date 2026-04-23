import cv2
import threading
import time

class VideoFileThread:
    def __init__(self, video_path, name='VideoFile', rotate=False):
        """
        video_path: 本機影片檔案路徑 (例如: 'test_video.mp4')
        """
        self.video_path = video_path
        self.name = name
        self.rotate = rotate
        
        # --- OpenCV 影片讀取設定 ---
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"[{self.name}] 錯誤：無法開啟影片檔案 {self.video_path}")
            
        # 自動獲取影片的原始 FPS 以控制讀取速度 (避免影片瞬間播完)
        # self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = 13
        if self.fps <= 0:
            self.fps = 30.0 # 若無法取得，給予預設值
        self.delay = 1.0 / self.fps

        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        
        # 啟動讀取執行緒
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        """讀取影像迴圈"""
        while self.running and self.cap.isOpened():
            start_time = time.time()
            
            # 1. 讀取影格
            ret, frame = self.cap.read()

            if ret:
                if self.rotate:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                
                # 2. 寫入共享變數
                with self.lock:
                    self.frame = frame
            else:
                print(f"[{self.name}] 影片讀取完畢或發生錯誤")
                self.running = False
                break

            # 3. 速度控制：模擬串流時間間隔
            elapsed = time.time() - start_time
            sleep_time = max(0.0, self.delay - elapsed)
            time.sleep(sleep_time)

    def update_fps(self, new_fps):
        """動態更新讀取速度 (FPS)"""
        self.fps = new_fps
        self.delay = 1.0 / self.fps
        print(f"[{self.name}] 讀取速度更新為: {self.fps} FPS (每幀間隔 {self.delay:.3f} 秒)")


    def read(self):
        """提供給 DetectThread 呼叫的介面"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        """停止執行緒並釋放資源"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        if self.cap.isOpened():
            self.cap.release()
        print(f"[{self.name}] 執行緒與資源已釋放")