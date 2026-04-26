import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# 讀取 CSV 檔案
# csv_filename = "hand_log.csv"
csv_filename = 'hand_log0.csv'
df = pd.read_csv(csv_filename)

# 將 DataFrame 寫入 Excel 檔案，以便後續分析
# df.to_excel("hand_log.xlsx", index=False)

# 儲存數據
frames = df['Frame'].tolist()
r_vel = df['R_Velocity'].tolist()
l_vel = df['L_Velocity'].tolist()

# 建立 4 個子圖，共用 x 軸 (Frame)
fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=False)
# 設定 X 軸刻度範圍與間隔
x_ticks = np.arange(0, max(frames) + 13, 13)
y_ticks = np.arange(0, 1300, 150)  # 設定 y 軸刻度範圍與間隔


# 第三個子圖：右手速度 (R_Velocity)
axes[0].plot(frames, r_vel, marker='o', markersize=3, color='green', label='R_Velocity')
axes[0].set_ylabel('R_Velocity')
axes[0].set_xlabel('Frame')
axes[0].legend(loc='upper right')
axes[0].grid(True)
axes[0].set_xticks(x_ticks)  # 設定 X 軸刻度
axes[0].set_yticks(y_ticks)  # 設定 Y 軸刻度


# 第四個子圖：左手速度 (L_Velocity)
axes[1].plot(frames, l_vel, marker='o', markersize=3, color='blue', label='L_Velocity')
axes[1].set_ylabel('L_Velocity')
axes[1].set_xlabel('Frame')
axes[1].legend(loc='upper right')
axes[1].grid(True)
axes[1].set_xticks(x_ticks)  # 設定 X 軸刻度
axes[1].set_yticks(y_ticks)  # 設定 Y 軸刻度

plt.tight_layout()
plt.show()