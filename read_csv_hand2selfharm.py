import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

cam_id = 1  # 0 或 1，決定讀取哪個腳踝的資料

# 讀取 CSV 檔案
csv_filename = f"hand_log{cam_id}.csv"

df = pd.read_csv(csv_filename)

# 將 DataFrame 寫入 Excel 檔案，以便後續分析
# df.to_excel("hand_log_angle.xlsx", index=False)
# 設定滑動窗口大小 (連續 frame 的範圍，例如 5 幀)
window_size = 10
# # 計算移動平均角度
df["R_AvgAgl"] = df["R_Angel"].rolling(window=window_size, min_periods=window_size).mean()  # 若 min_periods 設定為 1，會將收集到的做平均，而非連續 frame 
df["L_AvgAgl"] = df["L_Angel"].rolling(window=window_size, min_periods=window_size).mean() 

# 儲存數據
frames = df['Frame'].tolist()
r_vel = df['R_Velocity'].tolist()
r_avg_angle = df['R_AvgAgl'].tolist()
l_vel = df['L_Velocity'].tolist()
l_avg_angle = df['L_AvgAgl'].tolist()
shift_count = df['Count'].tolist()
harm_events = df['Self Harm'].tolist()


# 建立 4 個子圖，共用 x 軸 (Frame)
fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=False)
x_ticks = np.arange(0, max(frames) + 12, 12)

# 右手腕位移速度 (RH_Dist)
axes[0].plot(frames, r_vel, marker='o', markersize=3, color='green', label='R_Vel')
axes[0].set_ylabel('R_Vel')
axes[0].set_xlabel('Frame')
axes[0].legend(loc='upper right')
axes[0].grid(True)
axes[0].set_xticks(x_ticks)  # 設定 X 軸刻度
axes[0].axhline(y=450, color='red', linestyle='--')
# 取得 y 軸當前範圍
y_min, y_max = axes[0].get_ylim()
axes[0].set_yticks(np.arange(0, y_max + 50, 100))
# axes[0].set_ylim(0, 200)  # 限制y軸範圍

# 右手臂平均彎取角度 (RH_AvgAngle)
axes[1].plot(frames, r_avg_angle, marker='o', markersize=3, color='green', label='R_AvgAgl')
axes[1].set_ylabel('R_AvgAgl')
axes[1].set_xlabel('Frame')
axes[1].legend(loc='upper right')
axes[1].grid(True)
axes[1].set_xticks(x_ticks)  # 設定 X 軸刻度
axes[1].axhline(y=120, color='red', linestyle='--')
# 取得 y 軸當前範圍
angle_y_min, angle_y_max = axes[1].get_ylim()
axes[1].set_yticks(np.arange(0, angle_y_max + 30, 30))
# axes[0].set_ylim(0, 200)  # 限制y軸範圍

# 左手腕位移速度 (LH_Dist)
axes[2].plot(frames, l_vel, marker='o', markersize=3, color='blue', label='L_Vel')
axes[2].set_ylabel('L_Vel')
axes[2].set_xlabel('Frame')
axes[2].legend(loc='upper right')
axes[2].grid(True)
axes[2].set_xticks(x_ticks)  # 設定 X 軸刻度
axes[2].axhline(y=450, color='red', linestyle='--')
axes[2].set_yticks(np.arange(0, y_max + 50, 100))

# 左手臂平均彎取角度 (LH_AvgAngle)
axes[3].plot(frames, l_avg_angle, marker='o', markersize=3, color='sienna', label='L_AvgAgl')
axes[3].set_ylabel('L_AvgAgl')
axes[3].set_xlabel('Frame')
axes[3].legend(loc='upper right')
axes[3].grid(True)
axes[3].set_xticks(x_ticks)  # 設定 X 軸刻度
axes[3].axhline(y=120, color='red', linestyle='--') # 即時檢測 20
# 取得 y 軸當前範圍
# y_min, y_max = axes[3].get_ylim()
axes[3].set_yticks(np.arange(0, angle_y_max + 30, 30))  
# axes[5].set_ylim(0, 100)  # 限制y軸範圍

# count 次數 (shift_count)
axes[4].plot(frames, shift_count, marker='o', markersize=3, color='blueviolet', label='Count')
axes[4].set_ylabel('Count')
axes[4].set_xlabel('Frame')
axes[4].legend(loc='upper right')
axes[4].grid(True)
axes[4].set_xticks(x_ticks)  # 設定 X 軸刻度
axes[4].axhline(y=3, color='red', linestyle='--')
y_min, y_max_count = axes[4].get_ylim()
axes[4].set_yticks(np.arange(0, y_max_count + 1, 1))
# axes[6].set_ylim(0, 10)  # 限制y軸範圍

# Frame 對 Self Harm 使用 step() 繪製方波
axes[5].step(frames, harm_events, where='post', color='red', label='Self Harm')
axes[5].set_ylabel('Self Harm')
axes[5].set_xlabel('Frame')
axes[5].legend(loc='upper right')
axes[5].grid(True)
axes[5].set_xticks(x_ticks)  # 設定 X 軸刻度
# 如果 trigger_events 為 0 與 1，可設定 y 軸範圍，讓方波更明顯
axes[5].set_ylim(-0.1, 1.1)
axes[5].set_yticks([0, 1])

###########################################################################################
plt.tight_layout()
plt.show()

