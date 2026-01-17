import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

cam_id = 1  # 0 或 1，決定讀取哪個腳踝的資料

# 讀取 CSV 檔案
csv_filename = f"head_log{cam_id}.csv"

df = pd.read_csv(csv_filename)

# 將 DataFrame 寫入 Excel 檔案，以便後續分析
# df.to_excel("hand_log_angle.xlsx", index=False)

# 儲存數據
frames = df['Frame'].tolist()
leftAnkle_near_disL = df['LL_Dist'].tolist()
rightAnkle_near_disL = df['RL_Dist'].tolist()
leftAnkle_near_disR = df['LR_Dist'].tolist()
rightAnkle_near_disR = df['RR_Dist'].tolist()
near_ankle_state = df['nearState'].tolist()
nose_vel = df['nose_Velocity'].tolist()
shift_count = df['Count'].tolist()
trigger_events = df['Trigger Event'].tolist()
harm_events = df['Self Harm'].tolist()


# 建立 4 個子圖，共用 x 軸 (Frame)
fig, axes = plt.subplots(9, 1, figsize=(10, 12), sharex=False)
x_ticks = np.arange(0, max(frames) + 12, 12)

# 左腳踝接近左邊牆壁的距離 (LL_Dist)
axes[0].plot(frames, leftAnkle_near_disL, marker='o', markersize=3, color='green', label='L_near_AB_D')
axes[0].set_ylabel('L_near_AB_D')
axes[0].set_xlabel('Frame')
axes[0].legend(loc='upper right')
axes[0].grid(True)
axes[0].set_xticks(x_ticks)  # 設定 X 軸刻度
axes[0].axhline(y=60, color='red', linestyle='--')
# 取得 y 軸當前範圍
y_min, y_max = axes[0].get_ylim()
axes[0].set_yticks(np.arange(0, y_max + 45, 50))
# axes[0].set_ylim(0, 200)  # 限制y軸範圍

# 右腳踝接近左邊牆壁的距離 (RL_Dist)
axes[1].plot(frames, rightAnkle_near_disL, marker='o', markersize=3, color='blue', label='R_near_AB_D')
axes[1].set_ylabel('R_near_AB_D')
axes[1].set_xlabel('Frame')
axes[1].legend(loc='upper right')
axes[1].grid(True)
axes[1].set_xticks(x_ticks)  # 設定 X 軸刻度
axes[1].axhline(y=60, color='red', linestyle='--')
axes[1].set_yticks(np.arange(0, y_max + 45, 50))

# 左腳踝接近右邊牆壁的距離 (LR_Dist)
axes[2].plot(frames, leftAnkle_near_disR, marker='o', markersize=3, color='green', label='L_near_AC_D')
axes[2].set_ylabel('L_near_AC_D')
axes[2].set_xlabel('Frame')
axes[2].legend(loc='upper right')
axes[2].grid(True)
axes[2].set_xticks(x_ticks)  # 設定 X 軸刻度
axes[2].axhline(y=60, color='red', linestyle='--')
# 取得 y 軸當前範圍
# y_min, y_max = axes[2].get_ylim()
axes[2].set_yticks(np.arange(0, y_max + 45, 50))
# axes[0].set_ylim(0, 200)  # 限制y軸範圍

# 右腳踝接近右邊牆壁的距離 (RR_Dist)
axes[3].plot(frames, rightAnkle_near_disR, marker='o', markersize=3, color='blue', label='R_near_AC_D')
axes[3].set_ylabel('R_near_AC_D')
axes[3].set_xlabel('Frame')
axes[3].legend(loc='upper right')
axes[3].grid(True)
axes[3].set_xticks(x_ticks)  # 設定 X 軸刻度
axes[3].axhline(y=60, color='red', linestyle='--')
axes[3].set_yticks(np.arange(0, y_max + 45, 50))
# **設定 Y 軸刻度 (0~200，每 50 一格)**
# axes[1].set_yticks(range(0, 201, 50))
# 取得 y 軸當前範圍
# axes[1].set_ylim(0, 200)  # 限制y軸範圍

# Frame 對 NearState 使用 step() 繪製方波
axes[4].step(frames, near_ankle_state, where='post', color='black', label='nearState')
axes[4].set_ylabel('NearState')
axes[4].set_xlabel('Frame')
axes[4].legend(loc='upper right')
axes[4].grid(True)
axes[4].set_xticks(x_ticks)  # 設定 X 軸刻度
# 如果 trigger_events 為 0 與 1，可設定 y 軸範圍，讓方波更明顯
axes[4].set_ylim(-0.1, 1.1)
axes[4].set_yticks([0, 1])

# 鼻子位移速度 (nose_Velocity)
axes[5].plot(frames, nose_vel, marker='o', markersize=3, color='sienna', label='nose_Velocity')
axes[5].set_ylabel('nose_Velocity')
axes[5].set_xlabel('Frame')
axes[5].legend(loc='upper right')
axes[5].grid(True)
axes[5].set_xticks(x_ticks)  # 設定 X 軸刻度
axes[5].axhline(y=200, color='red', linestyle='--') # 即時檢測 20
# 取得 y 軸當前範圍
y_min, y_max = axes[3].get_ylim()
axes[5].set_yticks(np.arange(0, y_max + 10, 50))  
# axes[5].set_ylim(0, 100)  # 限制y軸範圍

# count 次數 (shift_count)
axes[6].plot(frames, shift_count, marker='o', markersize=3, color='blueviolet', label='Count')
axes[6].set_ylabel('Count')
axes[6].set_xlabel('Frame')
axes[6].legend(loc='upper right')
axes[6].grid(True)
axes[6].set_xticks(x_ticks)  # 設定 X 軸刻度
axes[6].axhline(y=3, color='red', linestyle='--')
y_min, y_max_count = axes[6].get_ylim()
axes[6].set_yticks(np.arange(0, y_max_count + 1, 1))
# axes[6].set_ylim(0, 10)  # 限制y軸範圍

# Frame 對 Trigger Event 使用 step() 繪製方波
axes[7].step(frames, trigger_events, where='post', color='black', label='Trigger Event')
axes[7].set_ylabel('Trigger Event')
axes[7].set_xlabel('Frame')
axes[7].legend(loc='upper right')
axes[7].grid(True)
axes[7].set_xticks(x_ticks)  # 設定 X 軸刻度
# 如果 trigger_events 為 0 與 1，可設定 y 軸範圍，讓方波更明顯
axes[7].set_ylim(-0.1, 1.1)
axes[7].set_yticks([0, 1])

# Frame 對 Self Harm 使用 step() 繪製方波
axes[8].step(frames, harm_events, where='post', color='red', label='Self Harm')
axes[8].set_ylabel('Self Harm')
axes[8].set_xlabel('Frame')
axes[8].legend(loc='upper right')
axes[8].grid(True)
axes[8].set_xticks(x_ticks)  # 設定 X 軸刻度
# 如果 trigger_events 為 0 與 1，可設定 y 軸範圍，讓方波更明顯
axes[8].set_ylim(-0.1, 1.1)
axes[8].set_yticks([0, 1])

###########################################################################################
plt.tight_layout()
plt.show()

