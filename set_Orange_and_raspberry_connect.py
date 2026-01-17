import paramiko
from scp import SCPClient
import sys
import json
import os

# ==========================================
# 1. 定義裝置設定檔 (Profile)
# ==========================================
PROFILES = {
    "RPI": {
        "user": "raspberrypi",
        "pass": "806806806",
        "remote_dir": "/home/raspberrypi/project",
        "python_path": "/home/raspberrypi/miniforge3/envs/rpi3912/bin/python",  # 樹莓派的 Conda
        "use_sudo": True
    },
    "OPI": {
        "user": "root",
        "pass": "orange",  # 請改成您的 Orange Pi 密碼
        "remote_dir": "/root/project",
        # [修改點 3] 必須指向我們剛建好的虛擬環境，不然會報錯！
        "python_path": "/root/aiot_env/bin/python",
        # "python_path": "/usr/bin/python3",  # Orange Pi 的系統 Python
        "use_sudo": False  # Root 登入不需要 sudo
    }
}

# ==========================================
# 2. 定義您的所有裝置 (混合清單)
# ==========================================
# 格式: "IP": {"type": "型號", "id": 鏡頭ID, "port": 傳輸Port}
TARGETS = {
    # 原有的樹莓派
    "192.168.50.43": {"type": "RPI", "id": 0, "port": 5555},
    "192.168.50.228": {"type": "RPI", "id": 0, "port": 5556},

    # 新加入的 Orange Pi
    # "192.168.50.74": {"type": "OPI", "id": 1, "port": 5557},  # 假設 Port 錯開
    # [修改點 1] 新加入的 Orange Pi -> 改用 Hostname
    "room-101-cam.local": {"type": "OPI", "id": 0, "port": 5557},
}


LOCAL_FILES = {
    "code": "send_main_v2.py",
    "config_template": "config_template.json"
}


# 動態產生 Service 內容
def generate_service_content(profile):
    return f"""[Unit]
Description=AIOT Camera Service
After=network-online.target
Wants=network-online.target

[Service]
User={profile['user']}
ExecStart={profile['python_path']} {profile['remote_dir']}/send_main_v2.py
WorkingDirectory={profile['remote_dir']}
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
Environment=PYTHONUNBUFFERED=1
Restart=always
RestartSec=5s

[Install]
WantedBy=multi-user.target
"""


def create_ssh_client(ip, user, pwd):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(ip, username=user, password=pwd, timeout=5)
        return client
    except Exception as e:
        print(f"❌ [連線失敗] {ip}: {e}")
        return None


def deploy_to_target(ip, info):
    dev_type = info["type"]
    profile = PROFILES.get(dev_type)

    if not profile:
        print(f"⚠️ 未知的裝置類型: {dev_type}")
        return

    print(f"\n🚀 正在部署至 [{dev_type}] {ip} (Cam:{info['id']}, Port:{info['port']})...")

    # 1. 準備 config.json
    try:
        with open(LOCAL_FILES["config_template"], 'r') as f:
            config_data = json.load(f)

        config_data['cam_id'] = info['id']
        config_data['port'] = info['port']
        # 注意: 這裡要確認 server ip 是否正確

        temp_config = f"config_temp_{ip}.json"
        with open(temp_config, 'w') as f:
            json.dump(config_data, f, indent=4)
    except Exception as e:
        print(f"⚠️ 設定檔錯誤: {e}")
        return

    # 2. 產生對應的 Service 檔內容
    service_content = generate_service_content(profile)
    temp_service = f"service_temp_{ip}.service"
    with open(temp_service, "w", encoding="utf-8") as f:
        f.write(service_content)

    # 3. SSH 連線與傳輸
    ssh = create_ssh_client(ip, profile["user"], profile["pass"])
    if not ssh: return

    try:
        # 建立資料夾
        ssh.exec_command(f"mkdir -p {profile['remote_dir']}")

        with SCPClient(ssh.get_transport()) as scp:
            scp.put(LOCAL_FILES["code"], f"{profile['remote_dir']}/{LOCAL_FILES['code']}")
            scp.put(temp_config, f"{profile['remote_dir']}/config.json")
            scp.put(temp_service, f"{profile['remote_dir']}/aiot.service")
            print(f"✅ 檔案上傳完成")

        # 4. 啟動服務 (區分 sudo)
        # 如果是 RPI (use_sudo=True)，加 sudo
        # 如果是 OPI (use_sudo=False)，直接執行
        pfx = f"echo {profile['pass']} | sudo -S " if profile["use_sudo"] else ""

        commands = [
            f"{pfx}mv {profile['remote_dir']}/aiot.service /etc/systemd/system/",
            f"{pfx}systemctl daemon-reload",
            f"{pfx}systemctl enable aiot.service",
            f"{pfx}systemctl restart aiot.service"
        ]

        for cmd in commands:
            stdin, stdout, stderr = ssh.exec_command(cmd)
            status = stdout.channel.recv_exit_status()
            if status != 0:
                print(f"⚠️ 指令警告: {stderr.read().decode().strip()}")

        print(f"🎉 {ip} 部署成功！")

    except Exception as e:
        print(f"❌ 部署失敗: {e}")
    finally:
        ssh.close()
        # 清理暫存檔
        if os.path.exists(temp_config): os.remove(temp_config)
        if os.path.exists(temp_service): os.remove(temp_service)


if __name__ == "__main__":
    print("=== 混合裝置群組部署工具 (RPi + OrangePi) ===")
    for ip, info in TARGETS.items():
        deploy_to_target(ip, info)