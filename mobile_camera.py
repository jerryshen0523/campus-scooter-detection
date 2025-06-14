#!/usr/bin/env python3
"""
Mobile Camera Integration for Campus Scooter Detection
手機攝影機整合模組

支援透過IP Webcam應用程式連接手機攝影機
"""

import cv2
import numpy as np
import requests
import time
from urllib.parse import urljoin

class MobileCameraStream:
    def __init__(self, ip_address, port=8080, username=None, password=None):
        """
        初始化手機攝影機串流

        Args:
            ip_address (str): 手機IP位址
            port (int): 埠號 (預設8080)
            username (str): 使用者名稱 (可選)
            password (str): 密碼 (可選)
        """
        self.ip_address = ip_address
        self.port = port
        self.username = username
        self.password = password

        # 構建URL
        self.base_url = f"http://{ip_address}:{port}"
        self.stream_url = urljoin(self.base_url, "/video")
        self.photo_url = urljoin(self.base_url, "/photo.jpg")
        self.status_url = urljoin(self.base_url, "/status.json")

        print(f"手機攝影機配置:")
        print(f"基礎URL: {self.base_url}")
        print(f"串流URL: {self.stream_url}")

    def test_connection(self):
        """
        測試與手機的連接

        Returns:
            bool: 連接是否成功
        """
        try:
            # 嘗試獲取狀態
            response = requests.get(self.status_url, timeout=5)
            if response.status_code == 200:
                print("✓ 手機攝影機連接成功")
                status_data = response.json()
                print(f"電池電量: {status_data.get('battery_level', 'N/A')}%")
                print(f"視頻格式: {status_data.get('video_format', 'N/A')}")
                return True
            else:
                print(f"✗ 連接失敗，狀態碼: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"✗ 連接錯誤: {e}")
            return False

    def get_frame(self):
        """
        獲取單張影像

        Returns:
            numpy.ndarray or None: 影像陣列
        """
        try:
            response = requests.get(self.photo_url, timeout=5)
            if response.status_code == 200:
                # 將響應內容轉換為numpy陣列
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                return frame
            else:
                print(f"獲取影像失敗，狀態碼: {response.status_code}")
                return None
        except Exception as e:
            print(f"獲取影像錯誤: {e}")
            return None

    def create_video_capture(self):
        """
        創建OpenCV VideoCapture物件

        Returns:
            cv2.VideoCapture: 視頻捕獲物件
        """
        return cv2.VideoCapture(self.stream_url)

    def stream_video(self, detection_callback=None):
        """
        串流視頻並可選地執行偵測

        Args:
            detection_callback (function): 偵測回調函數
        """
        cap = self.create_video_capture()

        if not cap.isOpened():
            print("無法開啟手機攝影機串流")
            return

        print("開始手機攝影機串流... 按 'q' 退出")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("無法讀取影像")
                    break

                # 執行偵測回調（如果提供）
                if detection_callback:
                    frame = detection_callback(frame)

                # 顯示影像
                cv2.imshow('Mobile Camera Stream', frame)

                # 檢查退出鍵
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\n串流被中斷")

        finally:
            cap.release()
            cv2.destroyAllWindows()

def find_mobile_camera():
    """
    自動搜尋區域網路中的手機攝影機

    Returns:
        list: 找到的攝影機IP位址列表
    """
    import socket
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # 獲取本機IP範圍
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    ip_parts = local_ip.split('.')
    base_ip = '.'.join(ip_parts[:3])

    print(f"搜尋IP範圍: {base_ip}.1-254:8080")

    found_cameras = []

    def check_ip(ip):
        try:
            url = f"http://{ip}:8080/status.json"
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return ip
        except:
            pass
        return None

    # 使用線程池並行檢查IP
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = []
        for i in range(1, 255):
            ip = f"{base_ip}.{i}"
            futures.append(executor.submit(check_ip, ip))

        for future in as_completed(futures):
            result = future.result()
            if result:
                found_cameras.append(result)
                print(f"✓ 找到手機攝影機: {result}")

    return found_cameras

def setup_mobile_camera():
    """
    引導使用者設定手機攝影機

    Returns:
        MobileCameraStream or None: 配置好的攝影機物件
    """
    print("=== 手機攝影機設定指南 ===")
    print("1. 在手機上安裝 'IP Webcam' 應用程式")
    print("2. 確保手機和電腦連接到同一個WiFi網路")
    print("3. 在應用程式中點擊 'Start Server'")
    print("4. 記下顯示的IP位址")
    print()

    # 自動搜尋
    print("正在自動搜尋手機攝影機...")
    cameras = find_mobile_camera()

    if cameras:
        print(f"\n找到 {len(cameras)} 個攝影機:")
        for i, ip in enumerate(cameras, 1):
            print(f"{i}. {ip}")

        while True:
            try:
                choice = input(f"\n請選擇攝影機 (1-{len(cameras)}) 或輸入自訂IP: ")

                if choice.isdigit() and 1 <= int(choice) <= len(cameras):
                    selected_ip = cameras[int(choice) - 1]
                    break
                else:
                    # 嘗試作為IP位址
                    selected_ip = choice
                    break
            except ValueError:
                print("請輸入有效的選項")
    else:
        print("\n未找到攝影機，請手動輸入IP位址:")
        selected_ip = input("手機IP位址: ").strip()

    # 創建攝影機物件並測試
    camera = MobileCameraStream(selected_ip)

    if camera.test_connection():
        print("\n✓ 手機攝影機設定完成!")
        return camera
    else:
        print("\n✗ 無法連接到手機攝影機")
        print("請檢查:")
        print("- 手機和電腦是否在同一網路")
        print("- IP Webcam應用程式是否正在運行")
        print("- IP位址是否正確")
        return None

def demo_mobile_detection():
    """
    手機攝影機偵測演示
    """
    # 設定手機攝影機
    camera = setup_mobile_camera()

    if camera is None:
        return

    # 簡單的動作偵測演示
    def motion_detection(frame):
        # 這裡可以整合滑板車偵測器
        # 暫時使用動作偵測作為演示
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 添加時間戳
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Mobile Camera: {timestamp}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, "Ready for Scooter Detection", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return frame

    # 開始串流
    camera.stream_video(detection_callback=motion_detection)

if __name__ == "__main__":
    demo_mobile_detection()
