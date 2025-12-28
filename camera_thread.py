import cv2
import threading
import time
import zmq
import numpy as np

class CameraThread:
    def __init__(self, port, name='NetCamera', rotate=False):
        """
        port: 本機 (NX) 開放給樹莓派連線的 Port
        """
        self.port = port
        self.name = name
        self.rotate = rotate
        
        # --- ZMQ 設定 (Server 模式: Bind) ---
        self.context = zmq.Context()
        self.subscriber = self.context.socket(zmq.SUB)
        
        # NX 監聽所有網卡 (0.0.0.0)，等待樹莓派連進來
        bind_addr = f"tcp://0.0.0.0:{self.port}"
        print(f"[{self.name}] 啟動網路接收模式，監聽位址: {bind_addr}")
        try:
            self.subscriber.bind(bind_addr)
        except zmq.ZMQError as e:
            print(f"[{self.name}] Bind 失敗 (Port可能被占用): {e}")

        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, '')
        
        # 設定 Poller 避免 recv 卡死
        self.poller = zmq.Poller()
        self.poller.register(self.subscriber, zmq.POLLIN)

        self.fps = 30 # 網路串流預設值
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        
        # 啟動接收執行緒
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        """接收影像迴圈"""
        while self.running:
            try:
                # 1. 檢查是否有資料 (Timeout 2秒)
                socks = dict(self.poller.poll(2000))
                
                if self.subscriber in socks and socks[self.subscriber] == zmq.POLLIN:
                    # 2. 接收並解碼
                    jpg_data = self.subscriber.recv()
                    np_arr = np.frombuffer(jpg_data, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    if frame is not None:
                        if self.rotate:
                            frame = cv2.rotate(frame, cv2.ROTATE_180)
                        
                        with self.lock:
                            self.frame = frame
                    else:
                        print(f"[{self.name}] 影像解碼失敗")
                else:
                    # Timeout: 表示樹莓派可能斷線或沒送資料
                    # print(f"[{self.name}] 等待樹莓派連線中...") 
                    # 為了避免 log 洗版，這裡可以註解掉，或每隔幾秒印一次
                    time.sleep(1)

            except Exception as e:
                print(f"[{self.name}] ZMQ 接收錯誤: {e}")
                time.sleep(1)

    def read(self):
        """提供給 DetectThread 呼叫的介面"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.subscriber.close()
        self.context.term()