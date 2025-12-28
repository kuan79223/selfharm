import cv2

def show_multiple_cameras(camera_ids):
    # 創建一個包含所有相機的視訊捕捉物件
    cameras = [cv2.VideoCapture(camera_id) for camera_id in camera_ids]
    
    if not all(camera.isOpened() for camera in cameras):
        print("其中一個或多個相機無法啟動！")
        return

    # 持續顯示多個相機的視訊流
    while True:
        frames = []
        for camera in cameras:
            ret, frame = camera.read()
            if ret:
                frames.append(frame)
            else:
                frames.append(None)
        
        # 顯示每個相機的畫面
        for i, frame in enumerate(frames):
            if frame is not None:
                cv2.imshow(f"Camera {camera_ids[i]}", frame)
            else:
                print(f"無法捕捉來自 Camera {camera_ids[i]} 的畫面")

        # 按下 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 關閉所有相機並退出
    for camera in cameras:
        camera.release()
    
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # 指定相機ID，通常是從 0 開始，依照您的設備配置可能會不同
    camera_ids = [0, 2]  # 假設有兩個相機，分別為 0 和 1
    show_multiple_cameras(camera_ids)
