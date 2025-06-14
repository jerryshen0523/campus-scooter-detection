#!/usr/bin/env python3
"""
Campus Scooter Detection System
使用手機攝影機進行校園滑板車即時偵測

Author: [Your Name]
Date: 2025-06-06
Course: EEB215A 電腦視覺與影像處理概論
"""

import cv2
import numpy as np
import argparse
import os
from datetime import datetime
import imutils

class ScooterDetector:
    def __init__(self, cascade_path, min_size=(30, 30), scale_factor=1.1, min_neighbors=5):
        """
        初始化滑板車偵測器

        Args:
            cascade_path (str): Haar cascade XML檔案路徑
            min_size (tuple): 最小偵測尺寸
            scale_factor (float): 縮放因子
            min_neighbors (int): 最小鄰居數
        """
        self.cascade_path = cascade_path
        self.min_size = min_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

        # 載入訓練好的Haar cascade分類器
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"找不到cascade檔案: {cascade_path}")

        self.scooter_cascade = cv2.CascadeClassifier(cascade_path)
        if self.scooter_cascade.empty():
            raise ValueError(f"無法載入cascade檔案: {cascade_path}")

        print(f"✓ 成功載入滑板車偵測器: {cascade_path}")

    def detect_scooters(self, frame):
        """
        在影像中偵測滑板車

        Args:
            frame: 輸入影像

        Returns:
            list: 偵測到的滑板車邊界框 [(x, y, w, h), ...]
        """
        # 轉換為灰階影像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 直方圖等化提升對比度
        gray = cv2.equalizeHist(gray)

        # 偵測滑板車
        scooters = self.scooter_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        return scooters

    def draw_detections(self, frame, scooters):
        """
        在影像上繪製偵測結果

        Args:
            frame: 輸入影像
            scooters: 偵測結果

        Returns:
            frame: 繪製後的影像
        """
        for (x, y, w, h) in scooters:
            # 繪製邊界框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 添加標籤
            label = "Scooter"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), (0, 255, 0), -1)
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # 顯示偵測統計
        count_text = f"Scooters detected: {len(scooters)}"
        cv2.putText(frame, count_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return frame

    def process_video_stream(self, source=0, output_file=None):
        """
        處理視頻流進行即時偵測

        Args:
            source: 視頻源 (0為預設攝影機, 或IP位址)
            output_file: 輸出視頻檔案路徑 (可選)
        """
        # 開啟視頻流
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise ValueError(f"無法開啟視頻源: {source}")

        # 設定視頻參數
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # 視頻寫入器設定
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = None
        if output_file:
            out = cv2.VideoWriter(output_file, fourcc, 20.0, (640, 480))

        print("開始即時偵測... 按 'q' 退出, 按 's' 截圖")

        frame_count = 0
        detection_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("無法讀取影像")
                    break

                frame_count += 1

                # 調整影像大小
                frame = imutils.resize(frame, width=640)

                # 偵測滑板車
                scooters = self.detect_scooters(frame)

                if len(scooters) > 0:
                    detection_count += 1

                # 繪製結果
                result_frame = self.draw_detections(frame, scooters)

                # 添加時間戳
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(result_frame, timestamp, (10, result_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # 顯示結果
                cv2.imshow('Campus Scooter Detection', result_frame)

                # 保存視頻
                if out is not None:
                    out.write(result_frame)

                # 鍵盤事件處理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 截圖保存
                    screenshot_name = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(screenshot_name, result_frame)
                    print(f"截圖已保存: {screenshot_name}")

        except KeyboardInterrupt:
            print("\n偵測被中斷")

        finally:
            # 清理資源
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()

            print(f"\n偵測統計:")
            print(f"總幀數: {frame_count}")
            print(f"偵測到滑板車的幀數: {detection_count}")
            print(f"偵測率: {detection_count/frame_count*100:.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Campus Scooter Detection System')
    parser.add_argument('--cascade', required=True, help='Haar cascade XML檔案路徑')
    parser.add_argument('--source', default=0, help='視頻源 (0為攝影機, 或IP位址)')
    parser.add_argument('--output', help='輸出視頻檔案路徑')
    parser.add_argument('--scale', type=float, default=1.1, help='縮放因子')
    parser.add_argument('--neighbors', type=int, default=5, help='最小鄰居數')

    args = parser.parse_args()

    try:
        # 創建偵測器
        detector = ScooterDetector(
            cascade_path=args.cascade,
            scale_factor=args.scale,
            min_neighbors=args.neighbors
        )

        # 開始偵測
        detector.process_video_stream(
            source=args.source if str(args.source).isdigit() else args.source,
            output_file=args.output
        )

    except Exception as e:
        print(f"錯誤: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
