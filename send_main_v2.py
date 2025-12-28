import cv2
import zmq
import time
import socket
import logging
import sys
import json
import os

# ==========================================
# 預設設定 (當 config.json 讀取失敗時使用)
# ==========================================
DEFAULT_CONFIG = {
    "nx_server_ip": "192.168.137.98", #
    "port": 5555,                     #
    "cam_id": 0,                      #
    "retry_interval": 5,
    "camera_settings": {
        "width": 640,
        "height": 480,
        "fps": 30,
        "jpeg_quality": 90            #
    }
}

# 設定 Log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [CAM_SERVICE] - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def load_config(file_path='config.json'):
    """讀取設定檔，若失敗則回傳預設值"""
    if not os.path.exists(file_path):
        logging.warning(f"找不到設定檔 {file_path}，將使用預設參數。")
        return DEFAULT_CONFIG
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            logging.info(f"成功讀取設定檔: {file_path}")
            return config
    except Exception as e:
        logging.error(f"讀取設定檔失敗 ({e})，將使用預設參數。")
        return DEFAULT_CONFIG

def init_zmq(config):
    context = zmq.Context()
    sender = context.socket(zmq.PUB)
    sender.setsockopt(zmq.SNDHWM, 100)
    
    ip = config.get("nx_server_ip", DEFAULT_CONFIG["nx_server_ip"])
    port = config.get("port", DEFAULT_CONFIG["port"])
    
    target_address = f"tcp://{ip}:{port}"
    logging.info(f"嘗試連線到 NX Server: {target_address} ...")
    sender.connect(target_address)
    return context, sender

def main():
    # 1. 載入設定
    config = load_config()
    cam_settings = config.get("camera_settings", DEFAULT_CONFIG["camera_settings"])
    cam_id = config.get("cam_id", DEFAULT_CONFIG["cam_id"])
    retry_interval = config.get("retry_interval", DEFAULT_CONFIG["retry_interval"])

    # 2. 初始化 ZMQ
    context, sender = init_zmq(config)
    cap = None

    try:
        while True:
            # 鏡頭斷線重連機制
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(cam_id)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_settings["width"])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_settings["height"])
                    cap.set(cv2.CAP_PROP_FPS, cam_settings["fps"])
                    logging.info("鏡頭初始化成功，開始傳輸影像...")
                else:
                    logging.warning(f"無法開啟鏡頭 (ID: {cam_id})，{retry_interval}秒後重試...")
                    time.sleep(retry_interval)
                    continue

            ret, frame = cap.read()
            if not ret:
                logging.warning("影像掉幀 (Frame Drop)，重啟鏡頭...")
                cap.release()
                cap = None
                time.sleep(1)
                continue

            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), cam_settings["jpeg_quality"]])
            
            if ret:
                try:
                    sender.send(buffer.tobytes(), flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass # 網路塞車時丟棄
            
            time.sleep(0.01)

    except KeyboardInterrupt:
        logging.info("程式終止")
    except Exception as e:
        logging.error(f"發生嚴重錯誤: {e}")
    finally:
        if cap: cap.release()
        sender.close()
        context.term()

if __name__ == "__main__":
    main()