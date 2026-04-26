import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================
# 模組設定：資料載入與分析區間
# ==========================================
csv_filename = 'hand_log0.csv'
try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    print(f"找不到檔案 {csv_filename}，請確認路徑。")
    exit()

start_frame = 26   # 區間起始 Frame
end_frame = 65    # 區間結束 Frame

# 資料篩選 (Data Filtering)
df_subset = df[(df['Frame'] >= start_frame) & (df['Frame'] <= end_frame)].copy()

# 若該區間無資料，進行基礎錯誤處理防護
if df_subset.empty:
    print(f"警告：區間 {start_frame} 到 {end_frame} 之間無資料。")
else:
    # [關鍵整合] 資料平移：將 Frame 減去起始值，確保 X 軸從 0 開始
    frames_normalized = (df_subset['Frame'] - start_frame).tolist()
    
    # 儲存繪圖數據 (Velocity)
    r_vel = df_subset['R_Velocity'].tolist()
    l_vel = df_subset['L_Velocity'].tolist()
    
    # 儲存繪圖數據 (Y 座標原始數值)
    r_wrist_y = df_subset['R_Wrist_Y'].tolist()
    r_elbow_y = df_subset['R_Elbow_Y'].tolist()
    l_wrist_y = df_subset['L_Wrist_Y'].tolist()
    l_elbow_y = df_subset['L_Elbow_Y'].tolist()

    r_angle = df_subset['R_Angel'].tolist()
    l_angle = df_subset['L_Angel'].tolist()

    # 建立 4 個子圖，共用平移後的 x 軸 (sharex=True)
    fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True)
    
    # 動態設定 X 軸刻度範圍與間隔 (根據正規化後的 frames 最大值)
    max_frame = max(frames_normalized) if frames_normalized else 0
    x_ticks = np.arange(0, max_frame + 13, 13)
    y_ticks_vel = np.arange(0, 1300, 150)  
    y_ticks_angle = np.arange(0, 181, 30)  # 角度刻度：0 到 180 度，每 30 度一刻

    # --------------------------------------------------------

    # 右手腕與手肘高度 (Y 座標)
    axes[0].plot(frames_normalized, r_wrist_y, marker='o', markersize=3, color='blue', label='R_Wrist_Y')
    axes[0].plot(frames_normalized, r_elbow_y, marker='x', markersize=3, color='orange', linestyle='--', label='R_Elbow_Y')
    axes[0].set_ylabel('R_Y-Coordinate')

    # 右手速度 (R_Velocity)
    axes[1].plot(frames_normalized, r_vel, marker='o', markersize=3, color='green', label='R_Velocity')
    axes[1].set_ylabel('R_Velocity')
    axes[1].set_yticks(y_ticks_vel)

    # 右手角度 (R_Angle)
    axes[2].plot(frames_normalized, r_angle, marker='s', markersize=3, color='brown', label='R_Angle')
    axes[2].set_ylabel('Angle (Degree)')
    axes[2].set_yticks(y_ticks_angle)

    # ---------------------------------------------------------
     # 左手腕與手肘高度 (Y 座標)
    axes[3].plot(frames_normalized, l_wrist_y, marker='o', markersize=3, color='cyan', label='L_Wrist_Y')
    axes[3].plot(frames_normalized, l_elbow_y, marker='x', markersize=3, color='magenta', linestyle='--', label='L_Elbow_Y')
    axes[3].set_ylabel('L_Y-Coordinate')

    # 左手速度 (L_Velocity)
    axes[4].plot(frames_normalized, l_vel, marker='o', markersize=3, color='purple', label='L_Velocity')
    axes[4].set_ylabel('L_Velocity')
    axes[4].set_yticks(y_ticks_vel)

    # 左手角度 (L_Angle)
    axes[5].plot(frames_normalized, l_angle, marker='s', markersize=3, color='brown', label='R_Angle')
    axes[5].set_ylabel('Angle (Degree)')
    axes[5].set_yticks(y_ticks_angle)

    # 統一設定 X 軸格式與標籤 (解決擠壓與未顯示問題)
    for ax in axes:
        ax.set_xticks(x_ticks)
        ax.tick_params(labelbottom=True)  # 強制每個圖表顯示數字
        ax.set_xlabel(f'Relative Frame (Offset = {start_frame})') # 標示為相對 Frame
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    # 調整垂直間距，避免文字被裁切
    plt.tight_layout(h_pad=3.0)
    plt.show()