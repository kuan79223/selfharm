import cv2
import socket
import struct
import numpy as np
import threading
import time

# --- 設定 ---
# RPI_IP = '192.168.50.43' # 請修改為樹莓派 IP
RPI_IP = '192.168.50.228' # 請修改為樹莓派 IP

PORT = 8000

# 用全域字典來存放最新畫面: {cam_id: image_array}
latest_frames = {}
lock = threading.Lock()
is_running = True

def recvall(sock, n):
    data = b''
    while len(data) < n:
        try:
            packet = sock.recv(n - len(data))
            if not packet: return None
            data += packet
        except socket.error:
            return None
    return data

def receiver_thread():
    global is_running, latest_frames
    
    while is_running:
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(5) # 設定超時，避免斷線卡死
            print(f"Connecting to {RPI_IP}...")
            client.connect((RPI_IP, PORT))
            print("Connected! Receiving stream...")
            
            payload_size = struct.calcsize("BL")

            while is_running:
                # 1. 收 Header
                header = recvall(client, payload_size)
                if not header: break # 斷線重連

                cam_id, msg_size = struct.unpack("BL", header)

                # 2. 收 Body
                frame_data = recvall(client, msg_size)
                if not frame_data: break

                # 3. 解碼
                np_data = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

                # 4. 更新最新畫面 (加鎖保護)
                with lock:
                    latest_frames[cam_id] = frame

        except Exception as e:
            print(f"Connection error: {e}. Retrying in 2s...")
            time.sleep(2)
        finally:
            client.close()

def main():
    global is_running
    
    # 啟動接收執行緒
    t = threading.Thread(target=receiver_thread)
    t.daemon = True # 設為守護執行緒，主程式結束它也會結束
    t.start()

    print("Display started. Press 'q' to exit.")
    
    # 等待直到收到第一幀資料
    while not latest_frames:
        time.sleep(0.1)

    while True:
        # 建立一個空白畫布 (寬度 640, 高度 240) 準備放兩張圖
        # 假設原始解析度 160x120，我們放大兩倍顯示
        canvas = np.zeros((240, 640, 3), dtype=np.uint8)
        
        with lock:
            # 複製字典，避免遍歷時被修改
            current_frames = latest_frames.copy()

        # 將畫面繪製到畫布上
        x_offset = 0
        for cam_id in sorted(current_frames.keys()):
            frame = current_frames[cam_id]
            if frame is not None:
                # 放大影像以利觀察
                resized = cv2.resize(frame, (320, 240))
                
                # 檢查是否超出畫布範圍
                if x_offset + 320 <= canvas.shape[1]:
                    canvas[0:240, x_offset:x_offset+320] = resized
                    
                    # 加上文字標記
                    cv2.putText(canvas, f"ID: {cam_id}", (x_offset+10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    x_offset += 320

        cv2.imshow("Dual Thermal Stream (NX)", canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_running = False
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()