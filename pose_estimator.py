from collections import deque
import math
import time
from queue import Queue
import numpy as np
import csv

import wall_info


# 三個點[肩膀 0 手肘 1 手腕 2]
# 三角形的餘弦定理
def getAngle(kpts):
    a = math.sqrt(((kpts[1][0] - kpts[2][0]) ** 2) + ((kpts[1][1] - kpts[2][1]) ** 2))
    b = math.sqrt(((kpts[0][0] - kpts[2][0]) ** 2) + ((kpts[0][1] - kpts[2][1]) ** 2))
    c = math.sqrt(((kpts[0][0] - kpts[1][0]) ** 2) + ((kpts[0][1] - kpts[1][1]) ** 2))
    try:
        B = int(math.degrees(math.acos((b * b - a * a - c * c) / (-2 * a * c))))
    except:
        B = 0
    return B

# 改寫判斷連續事件的方式
def check_consecutive_trigger_with_one_break(queue, required_consecutive=3):
    count = 0
    for ev in reversed(queue):  # 從最新的事件往前看
        if ev == 1:
            count += 1
            if count >= required_consecutive:
                return True
        else:
            break  # 一遇到0就中斷
    return False

# # 設定幾秒當作是觀察區間
# INTERVAL = 10


class PoseEstimator(object):

    def __init__(self, cf_thres, cam_id, scale_ratio, wall_config=None, record_diagram_to_csv=False, algo_cfg=None):
        
        if algo_cfg is None:
            algo_cfg = {}

        self.record_diagram_to_csv = record_diagram_to_csv # 是否儲存成csv檔
        self.cf_thres = cf_thres  # 信心值門檻
        self.cam_id = cam_id
        self.scale_ratio = scale_ratio  # 新增：縮放倍率
        self.wall_config = wall_config  # 接收牆壁設定
        self.time_interval = 0.0

        self.queue_size = None
        print('預設序列長度', self.queue_size)
        self.frame_queue = None
        # self.threshold_distance = 300  # 手腕位移速度門檻值  ****** 用兩顆鏡頭 FPS 會被減半，門檻值也減半ㄆ
        self.threshold_distance = algo_cfg.get('wrist_threshold_dist', 300)
        self.thres_avgAngle = algo_cfg.get('elbow_thres_avgAngle', 120)  # 手肘平均角度門檻
        base_ankle_dist = algo_cfg.get('ankle_thre_dist_base', 15)  # 腳踝接近牆壁的門檻值 15
        self.ankle_thre_dist = base_ankle_dist * scale_ratio
        self.head_thre_dist = algo_cfg.get('head_threshold_dist', 150)

        self.thres_count = None
        self.thre_velocity = 0
        self.ratio_fps = 0
        self.hand_shift_flag = 0  # 手腕位移速度條件狀態
        self.hand_shift_count = 0  # 計算滿足手腕位移條件的次數
        self.hand_trigger_event = 0
        self.current_timestamp = None
        self.csv_hand = f'hand_log{cam_id}.csv'  # 記錄右手數據的 CSV 檔案
        self.right_wrist = None
        self.left_wrist = None
        # **************************************
        self.right_distance = 0  # 右手腕移動距離
        self.right_velocity = 0.0  # 右手腕位移速度
        self.right_elbow_angle = 0
        self.right_avg_angle = 0  # 右手肘平均角度
        # **************************************
        self.left_distance = 0  # 左手腕移動距離
        self.left_velocity = 0.0  # 左手腕位移速度
        self.left_elbow_angle = 0
        self.left_avg_angle = 0  # 左手肘平均角度

        self.frame_data = dict()
        self.frame_data_copy = dict()
        # **********************************************************************************
        # self.thres_avgAngle = 120  # 手肘平均角度門檻
        # **********************************************************************************
        # *********************
        self.continues_queue = None
        # self.continues_queue = Queue(maxsize=self.queue_size)
        # *********************
        self.hand_harm_alert_sent = False # 表示是否剛剛偵測出新事件
        #######################################
        # self.ankle_thre_dist = 15 * scale_ratio  # 腳踝接近牆壁的門檻值 15 * 4 
        self.near_ankle_state = False
        #######################################
        # 左右腳踝接近'左邊牆壁'的距離
        self.leftAnkleToLeftWallDist = 0.0
        self.rightAnkleToLeftWallDist = 0.0
        # 左右腳踝接近'右邊牆壁'的距離
        self.leftAnkleToRightWallDist = 0.0
        self.rightAnkleToRightWallDist = 0.0
        #######################################
        self.csv_head = f'head_log{cam_id}.csv'
        # self.head_thre_dist = 150  # 頭部位移速度的門檻值 50

        self.nose = None
        self.nose_distance = 0  # 鼻子移動距離
        self.nose_velocity = 0.0  # 鼻子位移速度
        self.head_shift_flag = 0  # 鼻子位移速度條件狀態
        self.head_shift_count = 0  # 計算滿足鼻子位移條件的次數
        self.head_trigger_event = 0
        self.head_harm_alert_sent = False

        self.head_angle_change = 0.0
        self.head_avg_angle = 0
        
        if record_diagram_to_csv:
            # 初始化 CSV，寫入標頭
            with open(self.csv_hand, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Frame', 'R_Velocity', 'R_Angel', 'R_AvgAgl', 'L_Velocity', 'L_Angel', 'L_AvgAgl', 'Count', 'Self Harm']) # 手肘角度

            # 觀察鼻子與牆壁距離 / 頭部的位移速度
            with open(self.csv_head, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Frame', 'LL_Dist', 'RL_Dist', 'LR_Dist', 'RR_Dist', 'nearState', 'nose_Velocity', 'Count', 'Trigger Event', 'Self Harm']) # 幀, 左腳接近左，右腳接近左
    
    def update_fps(self, fps_size):
        """根據新的 fps 動態調整序列長度"""
        if fps_size:
            # if new_queue_size != self.queue_size:
            self.queue_size = fps_size * wall_info.INTERVAL
            self.frame_queue = Queue(maxsize=self.queue_size)
            self.continues_queue = Queue(maxsize=self.queue_size)
            print(f'更新fps: {fps_size}，新的序列長度: {self.queue_size}')
            self.ratio_fps = 6 / fps_size  # 以12為base 序列長度 作為比例換算
            print(f'手腕位移速度需要超過" {round(self.threshold_distance * self.ratio_fps)}')
            self.thres_count = round(self.queue_size * 0.5)  # 以序列長度的一半作為門檻值
            print(f'定義門檻次數{self.thres_count}')
            self.thre_velocity = round(self.threshold_distance * self.ratio_fps)  # 50 * 1
            print(f'位移速度比例換算: {self.thre_velocity}')

    def update_hand_info(self, kpts_list, frame_count):

        valid_kpts = [kpt for kpt in kpts_list if kpt[2] > 0.5]
        # print(f'骨幹列表{kpts_list}, 長度{len(kpts_list)}')
        self.current_timestamp = time.time()
        #########################################
        # 可以 Loss 掉兩個低於門檻值的關節點
        if len(valid_kpts) >= 15:
            self.near_ankle_state = False
            ######### 計算腳踝 ##################################################################################################################
            left_ankle, right_ankle = kpts_list[15][:2], kpts_list[16][:2]
            # 左牆壁線
            if self.wall_config:
                wall_center = self.wall_config['center']
                left_wall_end = self.wall_config['left']
                right_wall_end = self.wall_config['right']
                # 左右腳接近左側牆壁線
                self.leftAnkleToLeftWallDist = self.calculate_distance_point_to_segment(left_ankle, wall_center, left_wall_end)
                self.rightAnkleToLeftWallDist = self.calculate_distance_point_to_segment(right_ankle, wall_center, left_wall_end)
                # 左右腳接近右側牆壁線
                self.leftAnkleToRightWallDist = self.calculate_distance_point_to_segment(left_ankle, wall_center, right_wall_end)
                self.rightAnkleToRightWallDist = self.calculate_distance_point_to_segment(right_ankle, wall_center, right_wall_end)

                # ******************************************************************
                if (self.leftAnkleToLeftWallDist < self.ankle_thre_dist or 
                    self.rightAnkleToLeftWallDist < self.ankle_thre_dist or 
                    self.leftAnkleToRightWallDist < self.ankle_thre_dist or 
                    self.rightAnkleToRightWallDist < self.ankle_thre_dist):
                    self.near_ankle_state = True

                # 在實驗室環境下測試用
                # # 左右腳
                # self.min_leftAnkleNearLeftWallDist = self.calculate_distance_point_to_segment(left_ankle, wall_center,
                #                                                                               left_wall_end)
                # self.min_rightAnkleNearLeftWallDist = self.calculate_distance_point_to_segment(right_ankle, wall_center,
                #                                                                                left_wall_end)
                # # 右牆壁 (如果需要)
                # self.min_leftAnkleNearRightWallDist = self.calculate_distance_point_to_segment(left_ankle, wall_center,
                #                                                                                right_wall_end)
                # self.min_rightAnkleNearRightWallDist = self.calculate_distance_point_to_segment(right_ankle,
                #                                                                                 wall_center,
                #                                                                                 right_wall_end)

                # if self.min_leftAnkleNearLeftWallDist < self.ankle_thre_dist or self.min_rightAnkleNearLeftWallDist < self.ankle_thre_dist:
                #     self.near_ankle_state = True
                else:
                    self.near_ankle_state = False
            else:
                # Fallback if no config provided (optional, or keep 0)
                self.near_ankle_state = False

            #   ********************** 正式 ***********************
            # -----------------------
            self.nose = kpts_list[0][:2]
            # -----------------------
            #####################################################################################################################################
            left_shoulder, left_elbow, self.left_wrist = kpts_list[5][:2], kpts_list[7][:2], kpts_list[9][:2]
            right_shoulder, right_elbow, self.right_wrist = kpts_list[6][:2], kpts_list[8][:2], kpts_list[10][:2]

            # 計算手肘角度
            self.right_elbow_angle = getAngle([right_shoulder, right_elbow, self.right_wrist])
            self.left_elbow_angle = getAngle([left_shoulder, left_elbow, self.left_wrist])
            # print(f'右手肘角度{self.right_elbow_angle}; 左手肘角度{self.left_elbow_angle}')

            # 若隊列非空，計算當前幀與上一幀的位移速度
            if self.frame_queue:
                if not self.frame_queue.empty():
                    prev_frame = self.frame_queue.queue[-1]  # 取得最近的上一幀

                    prev_right_wrist = np.array(prev_frame.get("right_wrist", [0, 0]))  # 使用 .get 保證存在
                    prev_left_wrist = np.array(prev_frame.get("left_wrist", [0, 0]))
                    prev_nose = np.array(prev_frame.get('nose', [0, 0]))
                    prev_right_ankle = np.array(prev_frame.get("right_ankle", [0, 0]))
                    prev_left_ankle = np.array(prev_frame.get("left_ankle", [0, 0]))

                    prev_timestamp = prev_frame.get("timestamp", self.current_timestamp)
                    # prev_timestamp = self.frame_data["timestamp"]

                    # 手腕位移距離 ==============================================================================
                    self.right_distance = self.calculate_distance(prev_right_wrist, self.right_wrist)
                    self.left_distance = self.calculate_distance(prev_left_wrist, self.left_wrist)
                    # print(f'手腕位移右:{self.right_distance}; 左:{self.left_distance}')
                    self.nose_distance = self.calculate_distance(prev_nose, self.nose)
                    # 時間間隔: 兩個連續的時間戳非常接近，導致它們的差值幾乎為 0，甚至可能出現 0，0不能當被除數，防止出錯
                    self.time_interval = self.current_timestamp - prev_timestamp

                    if self.time_interval > 0.001:  # 避免除以 0 或極小值
                        # # 手腕位移速度
                        self.right_velocity = (self.right_distance / self.time_interval)
                        self.left_velocity = (self.left_distance / self.time_interval)
                        # print(f'手腕位移速度右:{self.right_velocity}; 左:{self.left_velocity}')
                        self.nose_velocity = self.nose_distance / self.time_interval
                    else:
                        self.right_velocity = 0
                        self.left_velocity = 0
                        self.nose_velocity = 0

                    # 判斷是否超過門檻 --------------------------------
                    is_fast_wrist = (self.threshold_distance * self.ratio_fps <= self.right_velocity) or (
                                self.threshold_distance * self.ratio_fps <= self.left_velocity)
                    # ================================================================================================
                    is_fast_head = self.nose_velocity >= self.head_thre_dist * self.ratio_fps
                    #########################################
                    # 手部狀態
                    if is_fast_wrist and self.is_wrist_higher_elbow(left_elbow, self.left_wrist, right_elbow,
                                                                    self.right_wrist) and self.is_ankle_lower_than_hip:
                        self.hand_shift_flag = 1
                        # print(f'手腕位移速度超過門檻值右:{self.right_velocity}; 左:{self.left_velocity}')
                    else:
                        self.hand_shift_flag = 0

                    # 頭部狀態
                    if self.near_ankle_state and is_fast_head:
                        self.head_shift_flag = 1
                        # print(f'鼻子位移速度超過門檻值:{self.nose_velocity}')
                    else:
                        self.head_shift_flag = 0

                # 正常處理當前幀資料
                self.frame_data = {
                    "frame_count": frame_count,
                    'timestamp': self.current_timestamp,
                    "right_wrist": self.right_wrist,
                    "left_wrist": self.left_wrist,
                    "right_elbow_angle": self.right_elbow_angle,
                    "left_elbow_angle": self.left_elbow_angle,
                    "nose": self.nose,
                    "hand_shift_flag": self.hand_shift_flag,
                    "head_shift_flag": self.head_shift_flag,
                    "right_ankle": right_ankle,
                    "left_ankle": left_ankle,
                    'right_velocity': self.right_velocity,
                    'left_velocity': self.left_velocity,
                }

        else:
            # print(f'Frame {frame_count} 無骨幹，使用前一幀數據進行補值')
            self.hand_shift_flag = 0
            self.head_shift_flag = 0
            self.hand_trigger_event = 0
            self.head_trigger_event = 0
            self.right_velocity = 0
            self.left_velocity = 0
            # 如果人消失了，警報狀態必須強制解除，否則會卡在 True
            self.near_ankle_state = False  # <--- 接近牆壁線狀態
            self.hand_harm_alert_sent = False
            self.head_harm_alert_sent = False
            # 補充可能缺失的字段 =================================================================
            if self.frame_data_copy:
                self.frame_data = self.frame_data_copy.copy()
                # 確保補值時將所有欄位補充完整，並防止影響位移計算
                self.frame_data["right_velocity"] = 0
                self.frame_data["left_velocity"] = 0
                self.frame_data["right_elbow_angle"] = self.frame_data.get("right_elbow_angle", 0)
                self.frame_data["left_elbow_angle"] = self.frame_data.get("left_elbow_angle", 0)
                self.frame_data["hand_shift_flag"] = 0
                self.frame_data["head_shift_flag"] = 0
                
            else:
                # 若沒有上一幀資料，給與預設值
                self.frame_data = {
                    "frame_count": frame_count,
                    "right_velocity": 0,
                    "left_velocity": 0,
                    "right_elbow_angle": 0,
                    "left_elbow_angle": 0,
                    "hand_shift_flag": 0,
                    "head_shift_flag": 0,
                    # --- 新增以下補值以避免 KeyError ---
                    "right_wrist": [0, 0], 
                    "left_wrist": [0, 0],
                    "nose": [0, 0],
                    "right_ankle": [0, 0],
                    "left_ankle": [0, 0]
                }

        # 將每一幀的數據備分 ##############################################################
        self.frame_data_copy = self.frame_data.copy()

        # ======== 用字典存放序列 ========
        # 若隊列已滿，移除最舊的幀
        if self.frame_queue:
            if self.frame_queue.full():
                self.frame_queue.get()
            ##############################################################
            self.frame_queue.put(self.frame_data)

        ##############################################################
        if self.frame_queue:
            if self.frame_queue.qsize() >= self.queue_size:

                # 從隊列中提取所有幀的手腕 Y 座標
                # right_wrist_y_list = [frame["right_wrist"][1] for frame in list(self.frame_queue.queue)]
                # left_wrist_y_list = [frame["left_wrist"][1] for frame in list(self.frame_queue.queue)]
                right_wrist_y_list = [frame.get("right_wrist", [0, 0])[1] for frame in list(self.frame_queue.queue)]
                left_wrist_y_list = [frame.get("left_wrist", [0, 0])[1] for frame in list(self.frame_queue.queue)]
                # 計算右手腕 Y 軸最大與最小差
                right_wrist_y_max = max(right_wrist_y_list)
                right_wrist_y_min = min(right_wrist_y_list)
                right_wrist_y_diff = right_wrist_y_max - right_wrist_y_min
                # print(f'計算右手腕 Y 軸最大與最小差{right_wrist_y_diff}')
                # 計算左手腕 Y 軸最大與最小差
                left_wrist_y_max = max(left_wrist_y_list)
                left_wrist_y_min = min(left_wrist_y_list)
                left_wrist_y_diff = left_wrist_y_max - left_wrist_y_min
                # print(f'計算左手腕 Y 軸最大與最小差{left_wrist_y_diff}')

                is_not_y_shift_differ_max = right_wrist_y_diff < 50 or left_wrist_y_diff < 50

                # 計算最近 N 幀內超過門檻的次數
                self.hand_shift_count = sum(frame["hand_shift_flag"] for frame in list(self.frame_queue.queue))
                self.head_shift_count = sum(frame["head_shift_flag"] for frame in list(self.frame_queue.queue))
                # 如果取不到 right_wrist，就回傳 [0, 0]，然後取 [1] (即 y=0)

                right_angles = [frame["right_elbow_angle"] for frame in list(self.frame_queue.queue)]
                left_angles = [frame["left_elbow_angle"] for frame in list(self.frame_queue.queue)]
                # ******* 平均手肘角度 *********
                self.right_avg_angle = np.mean(right_angles)
                self.left_avg_angle = np.mean(left_angles)

                # 手部判斷 ************************************************
                if (self.hand_shift_count >= self.thres_count) and (
                        (10 < self.right_avg_angle <= self.thres_avgAngle) or (
                        10 < self.left_avg_angle <= self.thres_avgAngle)) and is_not_y_shift_differ_max:
                    self.hand_trigger_event = 1
                else:
                    self.hand_trigger_event = 0

                # 頭部判斷 ************************************************
                if self.head_shift_count >= self.thres_count:
                    self.head_trigger_event = 1
                else:
                    self.head_trigger_event = 0

                if self.continues_queue.full():
                    self.continues_queue.get()

                self.continues_queue.put({
                    "hand": self.hand_trigger_event,
                    "head": self.head_trigger_event
                })

                # print(f'\n事件發生的序列: {list(self.continues_queue.queue)}')
                if self.continues_queue.qsize() >= self.queue_size:
                    event_list = list(self.continues_queue.queue)

                    hand_events = [e["hand"] for e in event_list[-round(self.queue_size):]]
                    head_events = [e["head"] for e in event_list[-round(self.queue_size):]]

                    if check_consecutive_trigger_with_one_break(hand_events, required_consecutive=3):
                        if not self.hand_harm_alert_sent:
                            self.hand_harm_alert_sent = True
                    else:
                        self.hand_harm_alert_sent = False

                    if check_consecutive_trigger_with_one_break(head_events, required_consecutive=3):
                        if not self.head_harm_alert_sent:
                            self.head_harm_alert_sent = True
                    else:
                        self.head_harm_alert_sent = False
                        
            if self.record_diagram_to_csv:
                # 記錄到 CSV 手腕位移距離與速度
                with open(self.csv_hand, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [frame_count, 
                        round(self.right_velocity),
                        self.right_elbow_angle,
                        self.right_avg_angle,  # 這些資料如果沒變化會是 0
                        round(self.left_velocity),
                        self.left_elbow_angle,
                        self.left_avg_angle,  # 這些資料如果沒變化會是 0
                        self.hand_shift_count,
                        self.hand_harm_alert_sent])
                
                # 記錄到 CSV 腳踝與牆壁距離與鼻子位移速度
                with open(self.csv_head, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([frame_count, 
                        round(self.leftAnkleToLeftWallDist), 
                        round(self.rightAnkleToLeftWallDist), 
                        round(self.leftAnkleToRightWallDist), 
                        round(self.rightAnkleToRightWallDist), 
                        self.near_ankle_state, 
                        self.nose_velocity, 
                        self.head_shift_count, 
                        self.head_trigger_event, 
                        self.head_harm_alert_sent])
        # 返回 18
        return (frame_count, round(self.right_velocity), self.right_elbow_angle, self.right_avg_angle,
                round(self.left_velocity), self.left_elbow_angle, self.left_avg_angle,
                self.hand_shift_count, self.hand_harm_alert_sent,
                self.near_ankle_state,
                round(self.leftAnkleToLeftWallDist, 2),
                round(self.rightAnkleToLeftWallDist, 2),
                round(self.leftAnkleToRightWallDist, 2),
                round(self.rightAnkleToRightWallDist, 2),
                round(self.nose_velocity), self.head_shift_count, self.head_harm_alert_sent, self.head_angle_change)
    

    # 計算向量長度
    def calculate_distance(self, prev_point, curr_point):
        # 確保 prev_point 和 curr_point 不是 None
        if prev_point is None or curr_point is None:
            return 0
        # 計算歐幾里得距離 (在 160x122 解析度下)
        D = np.linalg.norm(np.array(curr_point) - np.array(prev_point))
        return D  # 返回標準化後的 D

    # 用於計算腳踝
    def calculate_distance_wall(self, point1, point2):
        """ 計算兩點間的歐式距離 """
        return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

    # 腳踝到牆壁的距離
    def calculate_distance_point_to_segment(self, point, seg_start, seg_end):
        """ 計算點到線段的距離 """
        px, py = point
        x1, y1 = seg_start
        x2, y2 = seg_end

        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 & dy == 0:
            # 起點和終點是同一個點
            return np.hypot(px - x1, py - y1)
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))  # 限制 t 在 0~1 之間

        nearest_x = x1 + t * dx
        nearest_y = y1 + t * dy

        distance = np.hypot(px - nearest_x, py - nearest_y)
        return distance

    def is_wrist_higher_elbow(self, left_elbow, left_wrist, right_elbow, right_wrist):

        # 右手腕和右手肘的y座標
        right_wrist_y = right_wrist[1]
        right_elbow_y = right_elbow[1]

        # 左手腕和左手肘的y座標
        left_wrist_y = left_wrist[1]
        left_elbow_y = left_elbow[1]

        # 如果任一手腕比手肘低，返回 True
        return right_wrist_y <= right_elbow_y or left_wrist_y <= left_elbow_y

    def is_ankle_lower_than_hip(self, kpts_list):
        # 左腳踝與左髖骨
        left_ankle_y = kpts_list[15][1]  # 左腳踝y坐標
        left_hip_y = kpts_list[11][1]  # 左髖骨y坐標

        # 右腳踝與右髖骨
        right_ankle_y = kpts_list[16][1]  # 右腳踝y坐標
        right_hip_y = kpts_list[12][1]  # 右髖骨y坐標

        # 判斷左腳踝是否高於左髖骨
        if left_ankle_y > left_hip_y:
            left_ankle_is_lower = True
        else:
            left_ankle_is_lower = False

        # 判斷右腳踝是否高於右髖骨
        if right_ankle_y > right_hip_y:
            right_ankle_is_lower = True
        else:
            right_ankle_is_lower = False

        return left_ankle_is_lower and right_ankle_is_lower

    # 加在 PoseEstimator 類別的最後面
    def reset_memory(self):
        """重置所有計數器與狀態，防止警報卡死"""
        # print("PoseEstimator: 重置記憶狀態")
        self.frame_queue = Queue(maxsize=self.queue_size)
        self.continues_queue = Queue(maxsize=self.queue_size)
        
        self.hand_shift_count = 0
        self.head_shift_count = 0
        self.hand_trigger_event = 0
        self.head_trigger_event = 0
        
        self.hand_harm_alert_sent = False
        self.head_harm_alert_sent = False
        self.hand_shift_flag = 0
        self.head_shift_flag = 0
        
        # 清空複製的幀數據
        self.frame_data = dict()
        self.frame_data_copy = dict()
    




# AB左牆壁線
# left_wall_start = wall_info.left_wall_line[0]  # (60, 120)
# left_wall_end = wall_info.left_wall_line[-1]   # (100, 80)
# 左右腳ankle near AB wall dist
# self.min_leftAnkleNearLeftWallDist = self.calculate_distance_point_to_segment(left_ankle, left_wall_start, left_wall_end)
# self.min_rightAnkleNearLeftWallDist = self.calculate_distance_point_to_segment(right_ankle, left_wall_start, left_wall_end)

# if self.min_leftAnkleNearLeftWallDist < self.ankle_thre_dist or self.min_rightAnkleNearLeftWallDist < self.ankle_thre_dist:
#     self.near_ankle_state = True
# else:
#     self.near_ankle_state = False