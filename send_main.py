import cv2
import zmq
import time
import socket

# ==========================================
# 設定區
# ==========================================
# 這是您向機構申請的 NX 固定 IP (請修改這裡)
NX_SERVER_IP = "192.168.137.98"  
# Cam0 的 RPi 請設 5555, Cam1 的 RPi 請設 5556
PORT = 5555  
CAM_ID = 0 

def main():
    context = zmq.Context()
    # 使用 PUB (發布者) 模式
    sender = context.socket(zmq.PUB)
    
    # 【關鍵修改】RPi 主動連線到 NX (Connect)
    # 這樣 RPi 自己的 IP 變動也沒關係
    target_address = f"tcp://{NX_SERVER_IP}:{PORT}"
    print(f"嘗試連線到 NX Server: {target_address} ...")
    sender.connect(target_address)

    cap = cv2.VideoCapture(CAM_ID)
    # 降低解析度以確保乙太網路流暢度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("無法開啟鏡頭！")
        return

    print("開始傳輸影像...")
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # JPEG 壓縮 (品質 90)
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            
            if ret:
                sender.send(buffer.tobytes())
                frame_count += 1
                
                if frame_count % 60 == 0:
                    print(f"已發送 {frame_count} 幀 -> {target_address}")
            
            # 簡單的流量控制
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("程式停止")
    finally:
        cap.release()
        sender.close()
        context.term()

if __name__ == "__main__":
    main()