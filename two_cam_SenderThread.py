import cv2
import socket
import struct
import threading
import time

# --- 設定 ---
SERVER_IP = '0.0.0.0'
PORT = 8000
CAMERA_INDEXES = [0, 2]  # 根據實際 ls /dev/video* 修改

class CameraThread(threading.Thread):
    def __init__(self, cam_id, width=160, height=120):
        super().__init__()
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(cam_id)

        # --- 關鍵修改開始 ---
        # 強制設定為 MJPG 格式 (FourCC 編碼)
        # 注意：這行必須在設定寬高之前執行，效果最好
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        # --- 關鍵修改結束 ---

        # 設定低解析度與 FPS (熱影像通常不高)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 9) 

        # --- 加入這段檢查 ---
        real_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        real_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[Check] Cam {cam_id} Set: {width}x{height} | Actual: {real_w}x{real_h}")
        # -------------------
        
        # Debug: 檢查設定是否成功
        actual_fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
        print(f"[Info] Cam {cam_id} codec: {codec}")

        self.running = True
        self.frame = None
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            ret, img = self.cap.read()
            img = cv2.rotate(img, cv2.ROTATE_180)
            if ret:
                with self.lock:
                    self.frame = img
            else:
                # 若讀不到影像，稍微等待避免 CPU 飆高
                time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False
        self.cap.release()

def start_sender():
    # 1. 啟動相機執行緒
    threads = []
    for idx in CAMERA_INDEXES:
        t = CameraThread(idx)
        t.start()
        threads.append(t)
        print(f"[Info] Camera {idx} thread started.")

    # 2. 等待連線
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((SERVER_IP, PORT))
    server.listen(1)
    print(f"Waiting for NX connection on {PORT}...")

    conn, addr = server.accept()
    print(f"Connected to {addr}")

    try:
        while True:
            for t in threads:
                frame = t.get_frame()
                if frame is None:
                    continue

                # 編碼並發送
                ret, encoded_img = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ret: continue

                data = encoded_img.tobytes()
                
                # 封包格式: [ID 1byte] + [Length 4bytes] + [Data]
                header = struct.pack("BL", t.cam_id, len(data))
                
                try:
                    conn.sendall(header + data)
                except socket.error:
                    print("Connection lost.")
                    return # 斷線則跳出重連

            # 控制發送頻率，避免塞爆網路
            time.sleep(0.03) 

    except Exception as e:
        print(f"Error: {e}")
    finally:
        for t in threads:
            t.stop()
        server.close()
        print("Sender closed.")

if __name__ == "__main__":
    start_sender()