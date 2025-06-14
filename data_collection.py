#!/usr/bin/env python3
"""
Data Collection Tool for Campus Scooter Detection
校園滑板車資料收集工具

用於收集和整理訓練資料的工具
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import shutil
import json
from datetime import datetime
import sys

# 可選的 tkinter 導入，避免在無圖形界面環境中出錯
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
    print("警告: 無法導入 tkinter，GUI 功能將不可用")

class DataCollector:
    def __init__(self, project_dir="./"):
        """
        初始化資料收集器

        Args:
            project_dir (str): 專案目錄路徑
        """
        try:
            self.project_dir = Path(project_dir)
            self.data_dir = self.project_dir / "data"
            self.positive_dir = self.data_dir / "positive"
            self.negative_dir = self.data_dir / "negative"
            self.annotations_dir = self.data_dir / "annotations"
            self.trained_model_dir = self.data_dir / "trained_model"

            # 創建目錄
            for dir_path in [self.positive_dir, self.negative_dir, self.annotations_dir, self.trained_model_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)

            # 統計資訊
            self.stats = {
                "positive_images": 0,
                "negative_images": 0,
                "annotations": 0,
                "last_updated": None
            }

            self.update_stats()
            print(f"✓ 資料收集器初始化完成")
            print(f"專案目錄: {self.project_dir.absolute()}")
            
        except Exception as e:
            print(f"初始化失敗: {e}")
            raise

    def update_stats(self):
        """更新資料統計"""
        try:
            # 統計正樣本（避免重複計算同名檔案，不分大小寫）
            positive_patterns = ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"]
            positive_images = []
            seen_pos = set()
            for pattern in positive_patterns:
                for img in self.positive_dir.glob(pattern):
                    name_lower = img.name.lower()
                    if name_lower not in seen_pos:
                        positive_images.append(img)
                        seen_pos.add(name_lower)
            self.stats["positive_images"] = len(positive_images)

            # 統計負樣本（避免重複計算同名檔案，不分大小寫）
            negative_images = []
            seen_neg = set()
            for pattern in positive_patterns:
                for img in self.negative_dir.glob(pattern):
                    name_lower = img.name.lower()
                    if name_lower not in seen_neg:
                        negative_images.append(img)
                        seen_neg.add(name_lower)
            self.stats["negative_images"] = len(negative_images)

            # 統計標註檔案
            annotations = list(self.annotations_dir.glob("*.txt"))
            self.stats["annotations"] = len(annotations)

            self.stats["last_updated"] = datetime.now().isoformat()
            
        except Exception as e:
            print(f"更新統計時發生錯誤: {e}")

    def show_stats(self):
        """顯示資料統計"""
        self.update_stats()
        print("\n=== 校園滑板車偵測 - 資料統計 ===")
        print(f"正樣本影像: {self.stats['positive_images']}")
        print(f"負樣本影像: {self.stats['negative_images']}")
        print(f"標註檔案: {self.stats['annotations']}")
        print(f"最後更新: {self.stats['last_updated']}")

        # 檢查資料完整性
        missing_annotations = self.stats['positive_images'] - self.stats['annotations']
        if missing_annotations > 0:
            print(f"⚠️  缺少 {missing_annotations} 個標註檔案")
        
        # 建議訓練參數
        if self.stats['positive_images'] >= 200:
            suggested_pos = min(int(self.stats['positive_images'] * 0.9), 1800)
            suggested_neg = min(int(self.stats['negative_images'] * 0.9), 3600)
            print(f"\n📊 建議訓練參數:")
            print(f"   --num_pos {suggested_pos}")
            print(f"   --num_neg {suggested_neg}")
        else:
            print(f"\n⚠️  建議至少收集 200 張正樣本影像（當前: {self.stats['positive_images']}）")

        print()

    def collect_from_video(self, video_path, output_type="positive", 
                          frame_interval=30, max_frames=100):
        """
        從視頻中收集影像

        Args:
            video_path (str): 視頻檔案路徑
            output_type (str): 輸出類型 ("positive" 或 "negative")
            frame_interval (int): 擷取幀間隔
            max_frames (int): 最大擷取幀數
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"視頻檔案不存在: {video_path}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"無法開啟視頻檔案: {video_path}")

        try:
            # 獲取視頻資訊
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            print(f"視頻資訊: {frame_count} 幀, {fps:.2f} FPS, {duration:.2f} 秒")

            output_dir = self.positive_dir if output_type == "positive" else self.negative_dir
            video_name = video_path.stem

            current_frame = 0
            saved_count = 0

            print(f"從視頻收集{output_type}樣本: {video_path}")
            print(f"輸出目錄: {output_dir}")

            while True:
                ret, frame = cap.read()
                if not ret or saved_count >= max_frames:
                    break

                if current_frame % frame_interval == 0:
                    # 保存幀
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    filename = f"{video_name}_{timestamp}_{saved_count:04d}.jpg"
                    filepath = output_dir / filename

                    # 使用高質量參數保存影像
                    success = cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    if success:
                        saved_count += 1
                        print(f"保存: {filename}")
                    else:
                        print(f"保存失敗: {filename}")

                current_frame += 1

        finally:
            cap.release()

        print(f"✓ 從視頻中收集了 {saved_count} 張影像")
        self.update_stats()

    def collect_from_camera(self, output_type="positive", duration=60):
        """
        從攝影機即時收集影像

        Args:
            output_type (str): 輸出類型 ("positive" 或 "negative")
            duration (int): 收集持續時間（秒）
        """
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            # 嘗試其他攝影機索引
            for i in range(1, 4):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"使用攝影機索引: {i}")
                    break
            else:
                raise ValueError("無法開啟任何攝影機")

        # 設置攝影機參數
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        output_dir = self.positive_dir if output_type == "positive" else self.negative_dir

        print(f"開始從攝影機收集{output_type}樣本")
        print(f"持續時間: {duration}秒")
        print("按 's' 保存當前幀, 按 'q' 退出")

        start_time = datetime.now()
        saved_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("無法讀取攝影機畫面")
                    break

                # 顯示幀
                display_frame = frame.copy()
                
                # 添加資訊文字
                info_lines = [
                    f"Type: {output_type}",
                    f"Saved: {saved_count}",
                    "Press 's' to save, 'q' to quit"
                ]
                
                for i, line in enumerate(info_lines):
                    y_pos = 30 + i * 40
                    cv2.putText(display_frame, line, (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Data Collection', display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    # 保存當前幀
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    filename = f"camera_{output_type}_{timestamp}_{saved_count:04d}.jpg"
                    filepath = output_dir / filename

                    success = cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if success:
                        saved_count += 1
                        print(f"保存: {filename}")
                    else:
                        print(f"保存失敗: {filename}")

                elif key == ord('q'):
                    break

                # 檢查時間限制
                if (datetime.now() - start_time).seconds >= duration:
                    print(f"達到時間限制 {duration} 秒")
                    break

        except KeyboardInterrupt:
            print("\n用戶中斷收集")
        finally:
            cap.release()
            cv2.destroyAllWindows()

        print(f"✓ 從攝影機收集了 {saved_count} 張影像")
        self.update_stats()

    def import_images(self, source_dir, output_type="positive"):
        """
        從目錄導入影像

        Args:
            source_dir (str): 來源目錄
            output_type (str): 輸出類型 ("positive" 或 "negative")
        """
        source_path = Path(source_dir)
        output_dir = self.positive_dir if output_type == "positive" else self.negative_dir

        if not source_path.exists():
            raise ValueError(f"來源目錄不存在: {source_dir}")

        # 支援的影像格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        imported_count = 0

        for image_path in source_path.iterdir():
            if image_path.suffix.lower() in image_extensions:
                # 複製檔案
                dest_path = output_dir / image_path.name
                shutil.copy2(image_path, dest_path)
                imported_count += 1
                print(f"導入: {image_path.name}")

        print(f"✓ 導入了 {imported_count} 張{output_type}影像")
        self.update_stats()

    def create_annotation_template(self, image_name):
        """
        為影像創建標註模板（YOLO格式）

        Args:
            image_name (str): 影像檔案名稱
        """
        # 取得影像路徑
        image_path = self.positive_dir / image_name
        if not image_path.exists():
            print(f"找不到影像: {image_path}")
            return

        # 創建標註檔案
        annotation_name = image_path.stem + ".txt"
        annotation_path = self.annotations_dir / annotation_name

        # 讀取影像尺寸
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"無法讀取影像: {image_path}")
            return
            
        height, width = image.shape[:2]

        # 創建空標註檔案（YOLO格式註解）
        with open(annotation_path, 'w', encoding='utf-8') as f:
            f.write("# YOLO格式標註檔案 - 校園滑板車偵測\n")
            f.write("# 格式: class_id x_center y_center width height\n")
            f.write("# 所有座標都是相對於影像尺寸的比例 (0.0-1.0)\n")
            f.write(f"# 影像尺寸: {width}x{height}\n")
            f.write("# 滑板車類別ID: 0\n")
            f.write("#\n")
            f.write("# 範例: 0 0.5 0.5 0.3 0.4\n")
            f.write("# (表示在影像中心位置，寬30%高40%的滑板車)\n")
            f.write("#\n")
            f.write("# 請刪除註解行並添加實際標註\n")

        print(f"✓ 創建YOLO格式標註模板: {annotation_path}")
        print(f"請使用LabelImg（YOLO模式）或手動編輯此檔案添加標註")

    def create_classes_file(self):
        """
        創建類別定義檔案
        """
        classes_file = self.data_dir / "classes.txt"
        
        # 定義滑板車檢測的類別
        classes = ["scooter"]
        
        with open(classes_file, 'w', encoding='utf-8') as f:
            for class_name in classes:
                f.write(f"{class_name}\n")
        
        print(f"✓ 創建類別檔案: {classes_file}")
        return classes_file

    def convert_yolo_to_haar_format(self):
        """
        將YOLO格式標註轉換為Haar Cascade訓練格式
        """
        print("轉換YOLO格式標註為Haar Cascade訓練格式...")
        
        # 收集所有正樣本影像
        positive_images = []
        image_patterns = ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"]
        
        for pattern in image_patterns:
            positive_images.extend(list(self.positive_dir.glob(pattern)))
        
        haar_annotations = []
        
        for image_path in positive_images:
            annotation_path = self.annotations_dir / f"{image_path.stem}.txt"
            
            if annotation_path.exists():
                try:
                    # 讀取影像尺寸
                    image = cv2.imread(str(image_path))
                    if image is None:
                        continue
                    height, width = image.shape[:2];
                    
                    # 讀取YOLO標註
                    with open(annotation_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines();
                    
                    objects = [];
                    for line in lines:
                        line = line.strip();
                        if line and not line.startswith('#'):
                            parts = line.split();
                            if len(parts) >= 5:
                                try:
                                    class_id, x_center, y_center, w, h = map(float, parts[:5]);
                                    
                                    # 轉換YOLO座標到絕對像素座標
                                    x = int((x_center - w/2) * width);
                                    y = int((y_center - h/2) * height);
                                    w_abs = int(w * width);
                                    h_abs = int(h * height);
                                    
                                    # 確保座標在有效範圍內
                                    x = max(0, min(x, width - 1));
                                    y = max(0, min(y, height - 1));
                                    w_abs = max(1, min(w_abs, width - x));
                                    h_abs = max(1, min(h_abs, height - y));
                                    
                                    objects.append(f"{x} {y} {w_abs} {h_abs}");
                                    
                                except ValueError:
                                    continue;
                    
                    if objects:
                        # 相對路徑（從專案根目錄）
                        rel_path = image_path.relative_to(self.project_dir);
                        line_content = f"{rel_path} {len(objects)} " + " ".join(objects);
                        haar_annotations.append(line_content);
                        
                except Exception as e:
                    print(f"處理檔案時出錯 {image_path}: {e}");
                    continue;
        
        return haar_annotations;

    def create_positive_samples_list(self):
        """
        創建正樣本列表檔案（Haar Cascade 訓練用）
        """
        positive_list_file = self.data_dir / "positive_samples.txt"
        
        # 轉換YOLO標註為Haar格式
        haar_annotations = self.convert_yolo_to_haar_format();
        
        with open(positive_list_file, 'w', encoding='utf-8') as f:
            for annotation in haar_annotations:
                f.write(f"{annotation}\n");
        
        print(f"✓ 創建正樣本列表: {positive_list_file}");
        print(f"  包含 {len(haar_annotations)} 個有效標註");
        return positive_list_file;

    def create_negative_samples_list(self):
        """
        創建負樣本列表檔案（Haar Cascade 訓練用）
        """
        negative_list_file = self.data_dir / "negative_samples.txt"
        
        # 收集所有負樣本影像
        negative_images = []
        image_patterns = ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"]
        
        for pattern in image_patterns:
            negative_images.extend(list(self.negative_dir.glob(pattern)))
        
        with open(negative_list_file, 'w', encoding='utf-8') as f:
            for image_path in negative_images:
                # 相對路徑（從專案根目錄）
                rel_path = image_path.relative_to(self.project_dir)
                f.write(f"{rel_path}\n")
        
        print(f"✓ 創建負樣本列表: {negative_list_file}")
        print(f"  包含 {len(negative_images)} 張負樣本影像")
        return negative_list_file

    def validate_training_data(self):
        """
        驗證訓練資料的品質和完整性
        """
        print("=== 驗證訓練資料 ===")
        
        # 基本統計
        self.update_stats()
        print(f"正樣本影像: {self.stats['positive_images']}")
        print(f"負樣本影像: {self.stats['negative_images']}")
        print(f"標註檔案: {self.stats['annotations']}")
        
        # 檢查最低要求
        issues = []
        if self.stats['positive_images'] < 100:
            issues.append(f"正樣本數量不足（建議至少200張，當前{self.stats['positive_images']}張）")
        
        if self.stats['negative_images'] < 500:
            issues.append(f"負樣本數量不足（建議至少1000張，當前{self.stats['negative_images']}張）")
        
        if self.stats['annotations'] < self.stats['positive_images'] * 0.8:
            issues.append(f"標註檔案過少（應該接近正樣本數量）")
        
        # 驗證標註品質
        valid_annotations, invalid_files = self.validate_annotations()
        
        if invalid_files:
            issues.extend(invalid_files[:5])  # 只顯示前5個錯誤
        
        # 檢查影像品質
        quality_issues = self.check_image_quality()
        if quality_issues:
            issues.extend(quality_issues)
        
        # 報告結果
        if not issues:
            print("✅ 訓練資料驗證通過！")
            print("\n建議的訓練命令:")
            suggested_pos = min(int(self.stats['positive_images'] * 0.9), 1800)
            suggested_neg = min(int(self.stats['negative_images'] * 0.9), 3600)
            print(f"python train_cascade.py --num_pos {suggested_pos} --num_neg {suggested_neg}")
            return True
        else:
            print("❌ 發現以下問題:")
            for i, issue in enumerate(issues[:10], 1):
                print(f"  {i}. {issue}")
            if len(issues) > 10:
                print(f"  ... 還有 {len(issues)-10} 個問題")
            return False

    def check_image_quality(self):
        """
        檢查影像品質
        """
        issues = []
        
        # 檢查正樣本影像
        image_patterns = ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"]
        positive_images = []
        
        for pattern in image_patterns:
            positive_images.extend(list(self.positive_dir.glob(pattern)))
        
        small_images = 0
        corrupted_images = 0
        
        for image_path in positive_images[:50]:  # 只檢查前50張
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    corrupted_images += 1
                    continue
                    
                height, width = image.shape[:2]
                if width < 100 or height < 100:
                    small_images += 1
                    
            except Exception:
                corrupted_images += 1
        
        if small_images > 0:
            issues.append(f"發現 {small_images} 張尺寸過小的影像（建議至少100x100像素）")
        
        if corrupted_images > 0:
            issues.append(f"發現 {corrupted_images} 張損壞的影像檔案")
        
        return issues

    def prepare_haar_training_data(self):
        """
        準備 Haar Cascade 訓練資料
        """
        print("=== 準備 Haar Cascade 訓練資料 ===")
        
        # 驗證資料品質
        if not self.validate_training_data():
            print("\n❌ 資料驗證失敗，請先修正上述問題")
            return False
        
        # 創建樣本列表
        positive_file = self.create_positive_samples_list()
        negative_file = self.create_negative_samples_list()
        
        # 創建必要目錄
        vec_dir = self.data_dir / "vec"
        vec_dir.mkdir(exist_ok=True)
        
        cascade_dir = self.data_dir / "cascade"
        cascade_dir.mkdir(exist_ok=True)
        
        # 計算建議參數
        suggested_pos = min(int(self.stats['positive_images'] * 0.9), 1800)
        suggested_neg = min(int(self.stats['negative_images'] * 0.9), 3600)
        
        print(f"\n=== Haar Cascade 訓練指南 ===")
        print(f"📁 資料目錄已準備完成")
        print(f"   正樣本列表: {positive_file}")
        print(f"   負樣本列表: {negative_file}")
        print(f"   向量檔案目錄: {vec_dir}")
        print(f"   訓練輸出目錄: {cascade_dir}")
        
        print(f"\n🔧 建議的訓練命令:")
        print(f"1. 創建樣本向量檔案:")
        print(f"   opencv_createsamples -info {positive_file} -bg {negative_file}")
        print(f"   -vec {vec_dir}/samples.vec -w 24 -h 24")
        
        print(f"\n2. 訓練分類器:")
        print(f"   opencv_traincascade -data {cascade_dir}")
        print(f"   -vec {vec_dir}/samples.vec -bg {negative_file}")
        print(f"   -numPos {suggested_pos} -numNeg {suggested_neg}")
        print(f"   -w 24 -h 24 -numStages 20")
        
        print(f"\n📊 訓練資料統計:")
        print(f"   可用正樣本: {self.stats['positive_images']}")
        print(f"   可用負樣本: {self.stats['negative_images']}")
        print(f"   建議使用正樣本: {suggested_pos}")
        print(f"   建議使用負樣本: {suggested_neg}")
        
        print(f"\n⚠️  重要提醒:")
        print(f"   - 確保已安裝 OpenCV 命令列工具")
        print(f"   - 訓練過程可能需要數小時")
        print(f"   - 建議先用較少的樣本測試")
        print(f"   - 訓練完成後模型將保存在 {self.trained_model_dir}")
        
        return positive_file, negative_file

    def launch_labelimg(self, image_dir=None):
        """
        啟動LabelImg標註工具

        Args:
            image_dir (str): 要標註的影像目錄，預設為正樣本目錄
        """
        if image_dir is None:
            image_dir = str(self.positive_dir)

        # 確保目錄存在
        Path(image_dir).mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # 創建類別檔案
        classes_file = self.create_classes_file()

        try:
            import subprocess
            import sys
            
            # 嘗試不同的啟動方式
            commands = [
                ["labelImg", image_dir],
                ["python", "-m", "labelImg", image_dir],
                [sys.executable, "-m", "labelImg", image_dir]
            ]
            
            for i, cmd in enumerate(commands):
                try:
                    print(f"嘗試啟動方式 {i+1}: {' '.join(cmd)}")
                    subprocess.Popen(cmd, cwd=str(self.project_dir))
                    print(f"✓ 啟動LabelImg標註工具")
                    print(f"影像目錄: {image_dir}")
                    print(f"類別檔案: {classes_file}")
                    print(f"標註目錄: {self.annotations_dir}")
                    print("\n📝 LabelImg 設定指南:")
                    print("1. 在LabelImg中，點擊 'View' -> 'Auto Save mode' 啟用自動儲存")
                    print("2. 點擊左側的 'PascalVOC' 按鈕，切換到 'YOLO' 格式")
                    print("3. 點擊 'Change Save Dir' 設定標註儲存目錄為 data/annotations")
                    print("4. 點擊 'Change default saved annotation folder' 確認設定")
                    print("5. 開始標註滑板車物件（類別：scooter）")
                    print("\n💡 標註技巧:")
                    print("- 盡量框住完整的滑板車")
                    print("- 包含車輪、踏板和把手")
                    print("- 避免框住過多背景")
                    print("- 確保每張影像都有對應的 .txt 標註檔案")
                    return
                except (FileNotFoundError, PermissionError) as e:
                    print(f"啟動失敗: {e}")
                    continue
                    
            raise FileNotFoundError("無法找到或啟動 labelImg")
            
        except Exception as e:
            print("✗ 找不到LabelImg工具或啟動失敗")
            print("\n📦 安裝LabelImg:")
            print("pip install labelImg")
            print("或")
            print("conda install labelimg")
            print(f"\n❌ 錯誤詳情: {e}")
            
            # 提供手動標註的說明
            print("\n=== YOLO 手動標註格式 ===")
            print(f"📁 影像檔案位置: {image_dir}")
            print(f"📁 標註檔案位置: {self.annotations_dir}")
            print(f"📄 類別檔案位置: {classes_file}")
            print("📝 標註格式: class_id x_center y_center width height")
            print("   - class_id: 0 (滑板車)")
            print("   - 所有座標都是相對比例 (0.0-1.0)")
            print("   - 範例: 0 0.5 0.5 0.3 0.4")
            print("✅ 完成標註後執行: python data_collection.py --action validate")

    def validate_annotations(self):
        """
        驗證標註檔案的有效性（YOLO格式）
        """
        print("驗證 YOLO 格式標註檔案...")

        valid_count = 0
        invalid_files = []

        annotation_files = list(self.annotations_dir.glob("*.txt"))
        
        if not annotation_files:
            print("未找到任何標註檔案")
            return 0, []

        for annotation_file in annotation_files:
            try:
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                valid_annotations = 0
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 5:
                            try:
                                # 檢查YOLO格式
                                class_id = int(parts[0])
                                x, y, w, h = map(float, parts[1:5])

                                # 檢查座標範圍
                                if (0 <= x <= 1 and 0 <= y <= 1 and 
                                    0 < w <= 1 and 0 < h <= 1 and
                                    class_id == 0):  # 只接受滑板車類別
                                    valid_annotations += 1
                                else:
                                    invalid_files.append(f"{annotation_file.name}:行{line_num} - 座標超出範圍或類別錯誤")
                            except ValueError as e:
                                invalid_files.append(f"{annotation_file.name}:行{line_num} - 數值格式錯誤: {e}")
                        else:
                            invalid_files.append(f"{annotation_file.name}:行{line_num} - YOLO格式不正確")

                if valid_annotations > 0:
                    valid_count += 1
                elif annotation_file.stat().st_size > 0:  # 檔案不為空但沒有有效標註
                    invalid_files.append(f"{annotation_file.name} - 無有效YOLO標註")

            except Exception as e:
                invalid_files.append(f"{annotation_file.name}: {e}")

        print(f"✓ 有效標註檔案: {valid_count}/{len(annotation_files)}")
        if invalid_files:
            print(f"✗ 發現問題: {len(invalid_files)}")
            for file in invalid_files[:10]:  # 只顯示前10個
                print(f"  - {file}")
            if len(invalid_files) > 10:
                print(f"  ... 還有 {len(invalid_files)-10} 個問題")

        return valid_count, invalid_files

def main():
    parser = argparse.ArgumentParser(
        description='校園滑板車偵測 - 資料收集工具 v1.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例用法:
  查看資料統計:     python data_collection.py --action stats
  攝影機收集資料:   python data_collection.py --action camera --type positive --duration 120
  從視頻收集:       python data_collection.py --action video --source video.mp4 --type positive
  匯入影像:         python data_collection.py --action import --source /path/to/images --type positive
  啟動標註工具:     python data_collection.py --action annotate
  驗證標註:         python data_collection.py --action validate
  準備訓練資料:     python data_collection.py --action prepare_haar
  驗證訓練資料:     python data_collection.py --action validate_training

更多資訊請參考: README.md
        """
    )
    
    parser.add_argument('--project_dir', default='./', help='專案目錄路徑 (預設: ./)')
    parser.add_argument('--action', 
                       choices=['stats', 'camera', 'video', 'import', 'annotate', 
                               'validate', 'prepare_haar', 'validate_training'],
                       required=True, 
                       help='執行動作')
    parser.add_argument('--type', choices=['positive', 'negative'], default='positive',
                       help='資料類型 (預設: positive)')
    parser.add_argument('--source', help='來源檔案或目錄')
    parser.add_argument('--duration', type=int, default=60, help='攝影機收集持續時間（秒，預設: 60）')
    parser.add_argument('--interval', type=int, default=30, help='視頻幀擷取間隔 (預設: 30)')
    parser.add_argument('--max_frames', type=int, default=100, help='最大擷取幀數 (預設: 100)')

    args = parser.parse_args()

    try:
        collector = DataCollector(args.project_dir)

        if args.action == 'stats':
            collector.show_stats()

        elif args.action == 'camera':
            collector.collect_from_camera(args.type, args.duration)

        elif args.action == 'video':
            if not args.source:
                raise ValueError("視頻收集需要指定 --source 參數")
            collector.collect_from_video(args.source, args.type, 
                                       args.interval, args.max_frames)

        elif args.action == 'import':
            if not args.source:
                raise ValueError("導入影像需要指定 --source 參數")
            collector.import_images(args.source, args.type)

        elif args.action == 'annotate':
            collector.launch_labelimg()

        elif args.action == 'validate':
            collector.validate_annotations()

        elif args.action == 'prepare_haar':
            collector.prepare_haar_training_data()
            
        elif args.action == 'validate_training':
            collector.validate_training_data()

    except KeyboardInterrupt:
        print("\n程序被用戶中斷")
        return 0
    except Exception as e:
        print(f"✗ 錯誤: {e}")
        import traceback
        print("詳細錯誤資訊:")
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
