import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 讀取 CSV 檔案
csv_filename = 'hand_log0.csv'
try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    print(f"找不到檔案 {csv_filename}，請確認路徑。")
    exit()

# 儲存繪圖數據 (提取全影片的資料)
frames = df['Frame'].tolist()
r_vel = df['R_Velocity'].tolist()
l_vel = df['L_Velocity'].tolist()

# (來源：文件未載，推論：需讀取手腕與手肘的 Y 軸資料以新增圖表)
r_wrist_y = df['R_Wrist_Y'].tolist()
r_elbow_y = df['R_Elbow_Y'].tolist()
l_wrist_y = df['L_Wrist_Y'].tolist()
l_elbow_y = df['L_Elbow_Y'].tolist()

# 建立 4 個子圖，共用 x 軸 (sharex=True 以利垂直對齊對照時間點)
fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

# 設定 X 軸刻度範圍與間隔
max_frame = max(frames) if frames else 0
x_ticks = np.arange(0, max_frame + 13, 13)
y_ticks_vel = np.arange(0, 1300, 150)  # 設定速度的 Y 軸刻度範圍與間隔

# ==========================================
# 第一個子圖：右手速度 (R_Velocity)
axes[0].plot(frames, r_vel, marker='o', markersize=3, color='green', label='R_Velocity')
axes[0].set_ylabel('Velocity')
# axes[0].set_title('Right Hand: Velocity')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.5)
axes[0].set_xticks(x_ticks)  # 設定 X 軸刻度
axes[0].set_yticks(y_ticks_vel)
axes[0].set_xlabel('Frame')


# 第二個子圖：右手腕與手肘高度 (Y 座標原始數值)
axes[1].plot(frames, r_wrist_y, marker='o', markersize=3, color='blue', label='R_Wrist_Y')
axes[1].plot(frames, r_elbow_y, marker='x', markersize=3, color='orange', linestyle='--', label='R_Elbow_Y')
axes[1].set_ylabel('Y-Coordinate')
# axes[1].set_title('Right Hand: Wrist vs Elbow Height (Raw Pixel Y)')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.5)
axes[1].set_xlabel('Frame')
axes[1].set_xticks(x_ticks)  # 設定 X 軸刻度


# ==========================================
# 第三個子圖：左手速度 (L_Velocity)
axes[2].plot(frames, l_vel, marker='o', markersize=3, color='purple', label='L_Velocity')
axes[2].set_ylabel('Velocity')
# axes[2].set_title('Left Hand: Velocity')
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.5)
axes[2].set_xticks(x_ticks)  # 設定 X 軸刻度
axes[2].set_yticks(y_ticks_vel)
axes[2].set_xlabel('Frame')

# 第四個子圖：左手腕與手肘高度 (Y 座標原始數值)
axes[3].plot(frames, l_wrist_y, marker='o', markersize=3, color='cyan', label='L_Wrist_Y')
axes[3].plot(frames, l_elbow_y, marker='x', markersize=3, color='magenta', linestyle='--', label='L_Elbow_Y')
axes[3].set_ylabel('Y-Coordinate')
# axes[3].set_title('Left Hand: Wrist vs Elbow Height (Raw Pixel Y)')
axes[3].set_xlabel('Frame')
axes[3].legend(loc='upper right')
axes[3].grid(True, alpha=0.5)
axes[3].set_xticks(x_ticks)  # 設定 X 軸刻度    

# 統一設定所有子圖的 X 軸刻度
for ax in axes:
    ax.set_xticks(x_ticks)
    ax.tick_params(labelbottom=True)  
    ax.set_xlabel('Frame')


# [關鍵修改] 使用 h_pad 參數加大子圖之間的垂直間距，避免 Frame 被擠掉
# 數值可依據您的螢幕或圖表輸出大小微調 (建議值為 2.0 到 3.0)
plt.tight_layout(h_pad=3.0)
plt.show()