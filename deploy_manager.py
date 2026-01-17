import paramiko
from scp import SCPClient
import sys
import json
import os

# ==========================================
# 部署設定
# ==========================================
RPI_USER = "raspberrypi"  #
RPI_PASS = "806806806"    #

# 格式: "IP": (Cam_ID, Port)  Cam_ID 是要傳給遠端電腦給予 相機編號
TARGETS = {
    "192.168.50.43": (0, 8000),
    "192.168.50.228": (0, 8001),

}

# 遠端目錄設定
REMOTE_DIR = "/home/raspberrypi/project"
SERVICE_NAME = "aiot.service"

# 本地檔案清單
LOCAL_FILES = {
    "code": "send_main_v2.py",
    "service": "aiot.service",
    "config_template": "config_template.json"
}

def create_ssh_client(ip, user, pwd):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(ip, username=user, password=pwd, timeout=5)
        return client
    except Exception as e:
        print(f"❌ [連線失敗] {ip}: {e}")
        return None

def deploy_to_target(ip, cam_id, port):
    print(f"\n🚀 正在部署至 Cam {cam_id} (Port: {port}, IP: {ip})...")
    
    # 1. 準備 config.json
    try:
        with open(LOCAL_FILES["config_template"], 'r') as f:
            config_data = json.load(f)
        
        # 動態寫入 ID 與 Port
        config_data['cam_id'] = cam_id
        config_data['port'] = port
        
        temp_config = f"config_temp_{ip}.json"
        with open(temp_config, 'w') as f:
            json.dump(config_data, f, indent=4)
    except Exception as e:
        print(f"⚠️  設定檔處理錯誤: {e}")
        return

    ssh = create_ssh_client(ip, RPI_USER, RPI_PASS)
    if not ssh: return

    try:
        # 2. 建立遠端資料夾 (如果不存在)
        ssh.exec_command(f"mkdir -p {REMOTE_DIR}")

        # 3. 上傳檔案 (程式碼 + 設定檔 + Service檔)
        with SCPClient(ssh.get_transport()) as scp:
            # 上傳主程式
            scp.put(LOCAL_FILES["code"], f"{REMOTE_DIR}/{LOCAL_FILES['code']}")
            print(f"✅ 程式碼已更新")
            
            # 上傳 Config
            scp.put(temp_config, f"{REMOTE_DIR}/config.json")
            print(f"✅ 設定檔已更新 (ID:{cam_id}, Port:{port})")
            
            # 上傳 Service 檔 (先傳到家目錄，再搬進去)
            scp.put(LOCAL_FILES["service"], f"{REMOTE_DIR}/{SERVICE_NAME}")
            print(f"✅ Service 檔已上傳")

        # 4. 設定 Systemd 自動啟動 (使用 sudo)
        commands = [
            # 把 service 檔搬到系統目錄
            f"echo {RPI_PASS} | sudo -S mv {REMOTE_DIR}/{SERVICE_NAME} /etc/systemd/system/",
            # 重新載入設定
            f"echo {RPI_PASS} | sudo -S systemctl daemon-reload",
            # 啟用開機自啟
            f"echo {RPI_PASS} | sudo -S systemctl enable {SERVICE_NAME}",
            # 重啟服務讓新設定生效
            f"echo {RPI_PASS} | sudo -S systemctl restart {SERVICE_NAME}"
        ]

        for cmd in commands:
            stdin, stdout, stderr = ssh.exec_command(cmd)
            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                print(f"⚠️  指令執行警告: {stderr.read().decode().strip()}")
        
        print(f"🎉 部署成功！服務已重啟。")

    except Exception as e:
        print(f"❌ 部署失敗: {e}")
    finally:
        ssh.close()
        if os.path.exists(temp_config):
            os.remove(temp_config)

if __name__ == "__main__":
    print("=== AIOT 樹莓派群組一鍵部署工具 ===")
    # 修正原本的迴圈 unpacking 錯誤
    for ip, (cam_id, port) in TARGETS.items():
        deploy_to_target(ip, cam_id, port)
        
    print("\n=== 所有作業結束 ===")