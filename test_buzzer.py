import Jetson.GPIO as GPIO
import time

# ==========================================
# 模組介面定義 (API & Constants)
# ==========================================
# 採用 BOARD 編號模式 (對應 Jetson NX 40-pin 實體標號)
# 這裡以 Pin 12 (GPIO) 為例，請依實際硬體接線修改
BUZZER_PIN = 32

def setup_buzzer():
    """
    初始化 GPIO 模組邊界與狀態。
    (來源：文件未載，基於前文低準位觸發邏輯推論)
    """
    GPIO.setmode(GPIO.BOARD)
    # 預設輸出 HIGH (高準位)，確保蜂鳴器在初始化時處於關閉狀態，避免誤響
    GPIO.setup(BUZZER_PIN, GPIO.OUT, initial=GPIO.HIGH)
    print(f"[系統日誌] 蜂鳴器 GPIO {BUZZER_PIN} 初始化完成，狀態：OFF")

def trigger_buzzer(duration_sec):
    """
    觸發蜂鳴器發聲指定秒數。
    """
    try:
        print("[系統日誌] 蜂鳴器：ON (輸出低準位)")
        GPIO.output(BUZZER_PIN, GPIO.LOW)  # 低準位觸發發聲
        time.sleep(duration_sec)
        
        print("[系統日誌] 蜂鳴器：OFF (輸出高準位)")
        GPIO.output(BUZZER_PIN, GPIO.HIGH) # 高準位停止發聲
        
    except Exception as e:
        print(f"[錯誤處理] 執行期間發生例外狀況: {e}")
        # 發生錯誤時強制將腳位拉高，避免蜂鳴器長鳴
        GPIO.output(BUZZER_PIN, GPIO.HIGH) 

def main():
    try:
        setup_buzzer()
        
        # 測試點：連續響三次，每次 0.5 秒
        for i in range(3):
            print(f"--- 測試循環 {i+1}/3 ---")
            trigger_buzzer(0.5)
            time.sleep(0.5) # 間隔 0.5 秒
            
    except KeyboardInterrupt:
        print("\n[系統日誌] 收到使用者中斷訊號 (Ctrl+C)")
    finally:
        # 系統邊界清理：釋放 GPIO 資源
        GPIO.cleanup()
        print("[系統日誌] GPIO 資源已釋放，程式安全結束")

if __name__ == '__main__':
    main()