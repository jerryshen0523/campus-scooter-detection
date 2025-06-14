#!/usr/bin/env python3
"""
Campus Scooter Detection Demo
校園滑板車偵測演示程式

整合所有功能的演示腳本
"""

import cv2
import argparse
import sys
import os
from pathlib import Path

# 導入專案模組
try:
    from scooter_detector import ScooterDetector
    from mobile_camera import MobileCameraStream, setup_mobile_camera
except ImportError as e:
    print(f"導入錯誤: {e}")
    print("請確保所有專案檔案都在同一目錄中")
    sys.exit(1)

def demo_webcam_detection(cascade_path):
    """
    使用電腦攝影機進行滑板車偵測演示

    Args:
        cascade_path (str): Haar cascade XML檔案路徑
    """
    print("=== 電腦攝影機滑板車偵測演示 ===")

    try:
        detector = ScooterDetector(cascade_path)
        detector.process_video_stream(source=0)
    except Exception as e:
        print(f"演示失敗: {e}")

def demo_mobile_detection(cascade_path):
    """
    使用手機攝影機進行滑板車偵測演示

    Args:
        cascade_path (str): Haar cascade XML檔案路徑
    """
    print("=== 手機攝影機滑板車偵測演示 ===")

    try:
        # 設定手機攝影機
        camera = setup_mobile_camera()

        if camera is None:
            print("手機攝影機設定失敗")
            return

        # 創建偵測器
        detector = ScooterDetector(cascade_path)

        # 定義偵測回調函數
        def detection_callback(frame):
            scooters = detector.detect_scooters(frame)
            return detector.draw_detections(frame, scooters)

        # 開始手機攝影機串流偵測
        camera.stream_video(detection_callback=detection_callback)

    except Exception as e:
        print(f"演示失敗: {e}")

def demo_video_detection(cascade_path, video_path):
    """
    使用視頻檔案進行滑板車偵測演示

    Args:
        cascade_path (str): Haar cascade XML檔案路徑
        video_path (str): 視頻檔案路徑
    """
    print(f"=== 視頻檔案滑板車偵測演示: {video_path} ===")

    try:
        detector = ScooterDetector(cascade_path)
        detector.process_video_stream(source=video_path)
    except Exception as e:
        print(f"演示失敗: {e}")

def demo_image_detection(cascade_path, image_path):
    """
    使用單張影像進行滑板車偵測演示

    Args:
        cascade_path (str): Haar cascade XML檔案路徑
        image_path (str): 影像檔案路徑
    """
    print(f"=== 單張影像滑板車偵測演示: {image_path} ===")

    try:
        # 創建偵測器
        detector = ScooterDetector(cascade_path)

        # 讀取影像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"無法讀取影像: {image_path}")

        # 偵測滑板車
        scooters = detector.detect_scooters(image)

        # 繪製結果
        result_image = detector.draw_detections(image, scooters)

        # 顯示結果
        cv2.imshow('Scooter Detection Result', result_image)

        print(f"偵測到 {len(scooters)} 個滑板車")
        print("按任意鍵關閉視窗...")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存結果
        output_path = f"detection_result_{Path(image_path).stem}.jpg"
        cv2.imwrite(output_path, result_image)
        print(f"結果已保存: {output_path}")

    except Exception as e:
        print(f"演示失敗: {e}")

def show_project_info():
    """
    顯示專案資訊
    """
    print("=" * 60)
    print("校園滑板車偵測系統 Campus Scooter Detection System")
    print("=" * 60)
    print()
    print("課程: EEB215A 電腦視覺與影像處理概論")
    print("主題: 基於Haar級聯分類器的校園滑板車偵測")
    print()
    print("功能特色:")
    print("• 使用自訓練的Haar cascade分類器")
    print("• 支援電腦攝影機和手機攝影機")
    print("• 即時偵測和視頻錄製")
    print("• 完整的資料收集和標註工具")
    print("• 模組化程式設計")
    print()
    print("系統需求:")
    print("• Python 3.7+")
    print("• OpenCV 4.x")
    print("• 已訓練的Haar cascade模型")
    print()

def check_requirements():
    """
    檢查系統需求
    """
    print("檢查系統需求...")

    # 檢查Python版本
    import sys
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 7:
        print(f"✓ Python版本: {python_version.major}.{python_version.minor}")
    else:
        print(f"✗ Python版本過低: {python_version.major}.{python_version.minor}")
        print("需要Python 3.7或更高版本")
        return False

    # 檢查OpenCV
    try:
        import cv2
        print(f"✓ OpenCV版本: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV未安裝")
        print("請執行: pip install opencv-python")
        return False

    # 檢查其他依賴
    required_modules = ['numpy', 'pathlib']
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module}未安裝")
            return False

    print("✓ 所有系統需求已滿足")
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Campus Scooter Detection Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python demo.py --info                           # 顯示專案資訊
  python demo.py --check                          # 檢查系統需求
  python demo.py --webcam --cascade model.xml    # 電腦攝影機演示
  python demo.py --mobile --cascade model.xml    # 手機攝影機演示
  python demo.py --video video.mp4 --cascade model.xml    # 視頻檔案演示
  python demo.py --image image.jpg --cascade model.xml    # 單張影像演示
        """
    )

    # 功能選項
    parser.add_argument('--info', action='store_true', help='顯示專案資訊')
    parser.add_argument('--check', action='store_true', help='檢查系統需求')
    parser.add_argument('--webcam', action='store_true', help='電腦攝影機演示')
    parser.add_argument('--mobile', action='store_true', help='手機攝影機演示')
    parser.add_argument('--video', help='視頻檔案路徑')
    parser.add_argument('--image', help='影像檔案路徑')

    # 模型參數
    parser.add_argument('--cascade', help='Haar cascade XML檔案路徑')

    args = parser.parse_args()

    # 顯示專案資訊
    if args.info or len(sys.argv) == 1:
        show_project_info()
        return 0

    # 檢查系統需求
    if args.check:
        if check_requirements():
            return 0
        else:
            return 1

    # 檢查是否提供了cascade檔案
    if not args.cascade and (args.webcam or args.mobile or args.video or args.image):
        print("✗ 錯誤: 需要指定 --cascade 參數")
        print("如果還沒有訓練模型，請先執行:")
        print("  python train_cascade.py")
        return 1

    # 檢查cascade檔案是否存在
    if args.cascade and not os.path.exists(args.cascade):
        print(f"✗ 錯誤: 找不到cascade檔案: {args.cascade}")
        return 1

    try:
        # 執行相應的演示
        if args.webcam:
            demo_webcam_detection(args.cascade)

        elif args.mobile:
            demo_mobile_detection(args.cascade)

        elif args.video:
            if not os.path.exists(args.video):
                print(f"✗ 錯誤: 找不到視頻檔案: {args.video}")
                return 1
            demo_video_detection(args.cascade, args.video)

        elif args.image:
            if not os.path.exists(args.image):
                print(f"✗ 錯誤: 找不到影像檔案: {args.image}")
                return 1
            demo_image_detection(args.cascade, args.image)

        else:
            print("請指定一個演示選項 (--webcam, --mobile, --video, --image)")
            print("使用 --help 查看詳細說明")
            return 1

    except KeyboardInterrupt:
        print("\n演示被中斷")
    except Exception as e:
        print(f"✗ 演示失敗: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
