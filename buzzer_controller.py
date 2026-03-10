import threading

# --- 模組邊界與環境偵測 ---
# (來源：推論，跨平台硬體載入防呆機制，避免在開發機上崩潰)
try:
    import Jetson.GPIO as GPIO
    JETSON_ENV = True
except ImportError:
    JETSON_ENV = False

class BuzzerController:
    """
    非阻塞式蜂鳴器控制類別
    提供執行緒安全的硬體開關，並支援外部 Log 系統介接。
    """
    def __init__(self, pin=32, freq=267, duty_cycle=50, logger_callback=None):
        self.pin = pin
        self.freq = freq
        self.duty_cycle = duty_cycle
        self.pwm = None
        self.is_alarming = False
        self.lock = threading.Lock()
        self.timer = None # 新增: 計時器物件

        # 允許主程式將 sys_logger.log 傳入，解耦模組依賴
        self.logger_callback = logger_callback 
        self._setup()

    def _log(self, msg, is_error=False):
        """內部日誌轉發機制"""
        if self.logger_callback:
            self.logger_callback(msg, is_error)
        else:
            prefix = "[錯誤]" if is_error else "[資訊]"
            print(f"{prefix} BuzzerController: {msg}")

    def _setup(self):
        """初始化 GPIO 與硬體 PWM"""
        if not JETSON_ENV:
            self._log("未偵測到 Jetson.GPIO，蜂鳴器將以模擬模式運行。")
            return
        
        try:
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(self.pin, GPIO.OUT)
            self.pwm = GPIO.PWM(self.pin, self.freq)
            self._log(f"蜂鳴器硬體 (Pin {self.pin}) 初始化完成。")
        except Exception as e:
            self._log(f"蜂鳴器硬體初始化失敗: {e}", is_error=True)

    def start_alarm(self, duration_sec=None):
        """觸發持續警報音 (執行緒安全)，支援自動定時關閉"""
        with self.lock:
            if self.is_alarming:
                return # 若已在響，則忽略
            
            self.is_alarming = True
            if JETSON_ENV and self.pwm:
                try:
                    self.pwm.start(self.duty_cycle)
                except Exception as e:
                    self._log(f"硬體觸發異常: {e}", is_error=True)
            else:
                self._log(f"[硬體模擬] 🔊 蜂鳴器開始鳴叫！(Pin: {self.pin})")

            # 定時器邏輯
            if duration_sec is not None and duration_sec > 0:
                self.timer = threading.Timer(duration_sec, self.stop_alarm_from_timer)
                self.timer.start()
                self._log(f'已設定蜂鳴器於 {duration_sec} 秒後自動關閉。')
    
    # 由計時器觸發的內部停止方法
    def stop_alarm_from_timer(self):    
        self._log('定時器時間到，自動停止蜂鳴器。')
        self.stop_alarm()

    def stop_alarm(self):
        """停止警報音 (執行緒安全)"""
        with self.lock:
            # 若有計時器，手動停止時需取消計時器
            if self.timer:
                self.timer.cancel()
                self.timer = None
            
            if not self.is_alarming:
                return
            
            self.is_alarming = False
            if JETSON_ENV and self.pwm:
                try:
                    self.pwm.stop()
                except Exception as e:
                    self._log(f"硬體停止異常: {e}", is_error=True)
            else:
                self._log(f"[硬體模擬] 🔇 蜂鳴器已停止。")

    # 【新增】動態調整 PWM 工作週期 (音量)
    def set_volume(self, duty_cycle):
        """
        動態更新音量。若目前正在鳴叫，立即生效。
        :param duty_cycle: 0 ~ 50 (硬體層級的限制範圍)
        """
        with self.lock:
            # 確保不會超過 50，避免燒毀喇叭
            if duty_cycle > 50:
                duty_cycle = 50
            elif duty_cycle < 25:
                duty_cycle = 25
                
            self.duty_cycle = duty_cycle
            
            # 如果目前警報正在響，立即更改 PWM 輸出
            if self.is_alarming and JETSON_ENV and self.pwm:
                try:
                    self.pwm.ChangeDutyCycle(self.duty_cycle)
                except Exception as e:
                    self._log(f"調整音量異常: {e}", is_error=True)

    # 釋放模組資源(文件未載：依架構設計，
    # 全域 GPIO.cleanup() 應交由主程式統一執行，以免干擾其他腳位)
    def cleanup(self):
        self.stop_alarm()