import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 讀取 CSV 檔案
csv_filename = 'hand_log0.csv'
df = pd.read_csv(csv_filename)

# ==========================================
# 模組設定：定義分析區間 (修改這裡來調整要看的段落)
# ==========================================
start_frame = 37   # 區間起始 Frame
end_frame = 76    # 區間結束 Frame

# 資料篩選 (Data Filtering)
# 確保只取介於 start_frame 與 end_frame 之間的資料
df_subset = df[(df['Frame'] >= start_frame) & (df['Frame'] <= end_frame)].copy()

# 若該區間無資料，進行基礎錯誤處理防護
if df_subset.empty:
    print(f"警告：區間 {start_frame} 到 {end_frame} 之間無資料。")
else:
    # 資料平移：將 Frame 減去起始值，確保 X 軸從 0 開始
    frames_normalized = (df_subset['Frame'] - start_frame).tolist()
    
    # 儲存繪圖數據
    r_vel = df_subset['R_Velocity'].tolist()
    l_vel = df_subset['L_Velocity'].tolist()

    # 建立 2 個子圖，共用 x 軸 (Frame)
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=False)
    
    # 動態設定 X 軸刻度範圍與間隔 (根據正規化後的 frames 最大值)
    max_frame = max(frames_normalized) if frames_normalized else 0
    x_ticks = np.arange(0, max_frame + 13, 13)
    y_ticks = np.arange(0, 1300, 150)  

    # 第一個子圖：右手速度 (R_Velocity)
    axes[0].plot(frames_normalized, r_vel, marker='o', markersize=3, color='green', label='R_Velocity')
    axes[0].set_ylabel('R_Velocity')
    axes[0].set_xlabel(f'Relative Frame (Offset = {start_frame})') # 標示為相對 Frame
    axes[0].legend(loc='upper right')
    axes[0].grid(True)
    axes[0].set_xticks(x_ticks)  
    axes[0].set_yticks(y_ticks)  

    # 第二個子圖：左手速度 (L_Velocity)
    axes[1].plot(frames_normalized, l_vel, marker='o', markersize=3, color='blue', label='L_Velocity')
    axes[1].set_ylabel('L_Velocity')
    axes[1].set_xlabel(f'Relative Frame (Offset = {start_frame})') # 標示為相對 Frame
    axes[1].legend(loc='upper right')
    axes[1].grid(True)
    axes[1].set_xticks(x_ticks)  
    axes[1].set_yticks(y_ticks)  

    plt.tight_layout()
    plt.show()