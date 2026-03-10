import csv
import queue
from collections import deque

import cv2
import time
import json
import numpy as np

import onnxruntime
import torch
import torchvision
import torch.backends.cudnn as cudnn
from torchvision import transforms
from models.experimental import attempt_load

from .draw_human import DrawHumans
from .torch_utils import select_device
from .datasets import letterbox
from .general import non_max_suppression_kpt, output_to_keypoint
import poses_2



draw_scale = 0.3
draw_thinkness = 1
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
wrist_shift = poses_2.WristShiftCalculator(cf_thres=1, queue_size=8, fps=9)


class PoseEstimator(object):

    def __init__(self, weights='best.pt', imgsize=640, conf_thres=0.75, iou_thres=0.25):
        self.weights = weights
        self.imgsize = imgsize
        self.kpt_label = True

        # object confidence threshold
        self.conf_thres = conf_thres
        # IOU threshold for NMS
        self.iou_thres = iou_thres
        # Initialize
        self.device = select_device('')
        # Load model
        print('Load model...')
        load_start_time = time.time()
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        _ = self.model.eval()

        self.model.half()  # to FP16
        cudnn.benchmark = True
        # self.sess = onnxruntime.InferenceSession(self.weights, providers=['CUDAExecutionProvider'])
        loaded_model_time = time.time()
        print(f'Loaded complete: {loaded_model_time - load_start_time}')

        self.frame_count = 1

    # 計算整個流程的FPS
    def controller(self, frame):
        # start time for fps calculation
        start_time = time.time()
          # 這裡應該根據影像幀數更新

        img_rgb = self.preprocess(frame)  # 對影像進行預處理，轉換成適合模型輸入的格式。
        # print(f"Preprocessed Tensor Shape (Before Model Input): {img_rgb.shape}") # torch.Size([1, 3, 160, 122])

        im0, output = self.execute(img_rgb)  # 執行人體骨架偵測，獲取人體關鍵點資訊。

        # redefinf humans coordinates
        humans = self.estimate_relist(im0, output, self.imgsize)  # 解析並整理偵測結果，提取有用的關鍵點數據。

        # draw humans skeleton 在影像上繪製骨架，讓視覺化結果更清晰。
        im0 = DrawHumans.draw_skeleton(im0, humans)

        # 確保 im0 的解析度與原始 frame 相同
        if im0.shape[:2] != frame.shape[:2]:  # 檢查高寬是否相同
            im0 = cv2.resize(im0, (160, 122))  # 轉換回原解析度
        # ========================================================================

        # # humans 是陣列包字典，字典裡面會有人體骨幹，陣列是人體數量
        if humans:
        #     # 取第一個偵測到的人的關鍵點數據
            kpts_list = list(humans[0].values())
        #
        #     # 調用 `update_wrist_shift_cctv` 來計算手腕速度與角度變化
            right_dist, right_vel, left_dist, left_vel, count, trigger_event = wrist_shift.update_wrist_shift_cctv(kpts_list, self.frame_count)

            cv2.putText(im0, f"RWrist:Velocity={round(right_vel)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, draw_scale,
                                WHITE,
                                draw_thinkness, cv2.LINE_AA)
            cv2.putText(im0, f"LWrist:Velocity={round(left_vel)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, draw_scale,
                                WHITE,
                                draw_thinkness, cv2.LINE_AA)
            if trigger_event:
                cv2.putText(im0, "Self-Harm!", (100, 10), cv2.FONT_HERSHEY_SIMPLEX, draw_scale, WHITE,
                            draw_thinkness)
            else:
                cv2.putText(im0, " ", (100, 10), cv2.FONT_HERSHEY_SIMPLEX, draw_scale, WHITE, draw_thinkness)


        # Calculatio for FPS 計算 FPS，評估模型的運行速度。
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(im0, f"{round(fps)} ", (100, 10), cv2.FONT_HERSHEY_SIMPLEX, draw_scale, WHITE, draw_thinkness)

        self.frame_count += 1

        return im0, humans, fps

    def preprocess(self, frame):
        # convert frame to RGB
        orig_image = frame  # orig_image 保留原始影像 frame，避免直接修改原影像。

        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)  # 轉換影像格式 (BGR → RGB)  Opencv 轉成 Yolo 的顏色格式

        # letterbox() 是一種保持原始寬高比的影像縮放方法：
        # 參數分別為: (表示將影像調整為適當的大小。 auto=True 允許函式自動決定如何填充影像。stride=32 通常用於確保尺寸對齊 YOLO 的網格大小（32x32），提升模型效能。)
        # image = letterbox(image, (128, 128), auto=True, stride=32)[0]  # 應該是 (160, 122, 3)
        image = letterbox(image, (128, 128), auto=True, stride=32)[0]
        # print(f"After letterbox shape: {image.shape}")  # 應該是 (128, 160, 3)

        # 轉換影像為 PyTorch Tensor -------------------------------------q---
        # 將影像的數值範圍從 0-255 轉換為 0-1（即 uint8 → float32）。
        # 將影像的形狀從 (H, W, C) 變為 (C, H, W)，符合 PyTorch 的要求。
        image = transforms.ToTensor()(image)
        # image = image.unsqueeze(0)  # 增加 batch 維度  擇一

        #  轉換為 PyTorch 張量 ----------------------------------------
        # 主要是增加 batch 維度
        # image.numpy() 轉換為 NumPy 陣列 (C, H, W)。
        # np.array([image.numpy()]) 變成 (1, C, H, W)，符合神經網路的輸入要求。
        image = torch.tensor(np.array([image.numpy()])) # 增加 batch 維度

        # convert image data to device 移動到指定運算裝置（CPU / GPU）
        # self.device torch.device("cpu") 或 torch.device("cuda")
        image = image.to(self.device)
        # 檢查張量的形狀
        # print(f"Final Tensor Shape (Before Model Input): {image.shape}")  # 應該是 (1, 3, 128, 160)

        image = image.half() # uint8 to fp16/32 轉換為 FP16（半精度浮點數）
        # 只在 GPU 上使用 FP16
        # image = image.half() if self.device.type == 'cuda' else image.float()

        # 深度學習推論時，使用 FP16 可以降低記憶體佔用，提高計算效能（特別是 NVIDIA GPU）。
        # 如果 image.half() 遇到不支援 FP16 的情況，可以改用 image.float()（FP32）。
        return image

    def execute(self, image):
        # print(f"Executing Model with Input Shape: {image.shape}")
        # get predictions
        with torch.no_grad():
            output, _ = self.model(image, augment=False)
            # out = self.sess.run(['output'], {'images': image.numpy()})[0]
            # out = torch.from_numpy(out)

        # print(f"輸出模型內容: {output}")

        # Apply non max suppression
        output = non_max_suppression_kpt(output, self.conf_thres, self.iou_thres, nc=1, nkpt=17, kpt_label=True)
        output = output_to_keypoint(output)
        # print(f"Keypoint Output Shape: {output.shape}")  # 🔍 進一步檢查
        # print(f"Image tensor shape before conversion: {image.shape}")
        im0 = image[0].permute(1, 2, 0) * 255
        im0 = im0.cpu().numpy().astype(np.uint8)
        # reshape image format to (BGR)
        im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
        # print(f"Converted im0 shape: {im0.shape}")

        return im0, output

    def estimate_relist(self, image, output, imgsize):
        re_nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        humans = []
        steps = 3  # 每個關鍵點的資料量（x, y, confidence）
        for idx in range(output.shape[0]):  # 遍歷每個偵測到的人
            centers = {}
            exist_key = 0
            kpts = output[idx, 7:].T  # 取得關鍵點資訊
            num_kpts = len(kpts) // steps
            for kid in range(num_kpts):
                x_coord = int(kpts[steps * kid])
                y_coord = int(kpts[steps * kid + 1])
                conf = kpts[steps * kid + 2]  # 取得置信度

                if not (x_coord % imgsize == 0 or y_coord % imgsize == 0):  # 過濾掉邊界錯誤點

                    if conf < 0.5:  # 如果置信度低於 0.5
                        centers[re_nums[kid]] = (0, 0, 0.0)  # 設定 (x, y, conf) 為 (0, 0, 0.0)
                    else:
                        exist_key += 1
                        centers[re_nums[kid]] = (x_coord, y_coord, conf)  # 儲存 (x, y, confidence)
                else:
                    centers[re_nums[kid]] = (0, 0, 0.0)  # 邊界外的點設為 (0, 0, 0.0)

            if exist_key >= 7:  # 只儲存至少有 7 個有效關鍵點的人體
                # centers[1] = self.add_neck(centers)
                humans.append(centers)

        return humans

