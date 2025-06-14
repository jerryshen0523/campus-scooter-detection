#!/usr/bin/env python3
"""
Haar Cascade Training Script for Scooter Detection
滑板車Haar級聯分類器訓練腳本

此腳本自動化Haar cascade訓練過程
"""

import os
import subprocess
import argparse
import shutil
import stat
import time
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

class HaarCascadeTrainer:
    def __init__(self, project_dir="./"):
        """
        初始化Haar cascade訓練器

        Args:
            project_dir (str): 專案目錄路徑
        """
        self.project_dir = Path(project_dir).resolve()
        self.data_dir = self.project_dir / "data"
        self.positive_dir = self.data_dir / "positive"
        self.negative_dir = self.data_dir / "negative"
        self.annotations_dir = self.data_dir / "annotations"
        self.output_dir = self.data_dir / "trained_model"

        # 創建必要目錄
        for dir_path in [self.positive_dir, self.negative_dir, 
                        self.annotations_dir, self.output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f"訓練器初始化完成")
        print(f"專案目錄: {self.project_dir}")
        print(f"正樣本目錄: {self.positive_dir}")
        print(f"負樣本目錄: {self.negative_dir}")
        print(f"標註目錄: {self.annotations_dir}")
        print(f"輸出目錄: {self.output_dir}")

    def prepare_positive_samples(self):
        """
        準備正樣本描述檔案
        從標註檔案生成正樣本資訊

        Returns:
            str: 正樣本描述檔案路徑
        """
        positive_list_file = self.data_dir / "positive_images.txt"

        # 獲取所有正樣本影像和對應的標註
        positive_data = []
        invalid_annotations = []
        processed_files = set()  # 防止重複處理

        for img_ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']:
            for img_path in self.positive_dir.glob(img_ext):
                # 防止重複處理相同檔案
                if img_path.name in processed_files:
                    continue
                processed_files.add(img_path.name)
                
                # 尋找對應的標註檔案
                annotation_file = self.annotations_dir / f"{img_path.stem}.txt"

                if annotation_file.exists():
                    # 先讀取影像來獲取尺寸
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"警告: 無法讀取影像 {img_path}")
                        continue
                    
                    img_height, img_width = img.shape[:2]
                    print(f"處理影像 {img_path.name}: {img_width}x{img_height}")
                    
                    # 讀取標註資訊
                    annotations = []
                    
                    try:
                        with open(annotation_file, 'r', encoding='utf-8') as f:
                            for line_num, line in enumerate(f, 1):
                                line = line.strip()
                                if line and not line.startswith('#'):
                                    parts = line.split()
                                    if len(parts) >= 5:  # class x y w h (YOLO format)
                                        try:
                                            class_id, x_center, y_center, width, height = map(float, parts[:5])
                                            
                                            # 檢查YOLO格式座標範圍
                                            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                                                   0 < width <= 1 and 0 < height <= 1):
                                                invalid_annotations.append(f"{annotation_file.name}:行{line_num} - YOLO座標超出範圍")
                                                continue

                                            # 轉換為絕對座標，確保整數運算
                                            x_left = (x_center - width/2) * img_width
                                            y_top = (y_center - height/2) * img_height
                                            box_w = width * img_width
                                            box_h = height * img_height
                                            
                                            # 轉為整數並確保邊界
                                            x = max(0, int(x_left))
                                            y = max(0, int(y_top))
                                            w = max(10, min(int(box_w), img_width - x))
                                            h = max(10, min(int(box_h), img_height - y))
                                            
                                            # 再次檢查邊界，確保不會超出影像範圍
                                            if x + w > img_width:
                                                w = img_width - x
                                            if y + h > img_height:
                                                h = img_height - y
                                            
                                            # 確保最小尺寸和有效性
                                            if (w >= 10 and h >= 10 and 
                                                x >= 0 and y >= 0 and 
                                                x + w <= img_width and 
                                                y + h <= img_height):
                                                annotations.append((x, y, w, h))
                                                print(f"  標註 {line_num}: ({x}, {y}, {w}, {h})")
                                            else:
                                                invalid_annotations.append(f"{annotation_file.name}:行{line_num} - 邊界框無效: ({x}, {y}, {w}, {h}) vs 影像尺寸 ({img_width}, {img_height})")
                                                
                                        except ValueError as e:
                                            invalid_annotations.append(f"{annotation_file.name}:行{line_num} - 數值格式錯誤: {e}")
                                    
                    except Exception as e:
                        print(f"讀取標註檔案時出錯 {annotation_file}: {e}")
                        continue

                    if annotations:
                        positive_data.append((img_path, annotations))

        if not positive_data:
            raise ValueError("找不到有效的正樣本標註資料")

        if invalid_annotations:
            print(f"⚠️  發現 {len(invalid_annotations)} 個無效標註:")
            for error in invalid_annotations[:5]:  # 只顯示前5個
                print(f"   {error}")
            if len(invalid_annotations) > 5:
                print(f"   ... 還有 {len(invalid_annotations)-5} 個錯誤")

        # 寫入正樣本描述檔案，使用相對路徑
        with open(positive_list_file, 'w') as f:
            for img_path, annotations in positive_data:
                # 使用相對於專案根目錄的路徑，確保格式正確
                rel_path = str(img_path.relative_to(self.project_dir)).replace('\\', '/')
                line = f"{rel_path} {len(annotations)}"
                for x, y, w, h in annotations:
                    line += f" {x} {y} {w} {h}"
                f.write(line + "\n")

        print(f"✓ 準備了 {len(positive_data)} 個正樣本影像")
        total_objects = sum(len(annotations) for _, annotations in positive_data)
        print(f"總共 {total_objects} 個滑板車標註")
        print(f"正樣本列表: {positive_list_file}")
        
        # 顯示前幾行內容以便調試
        print("正樣本列表檔案內容預覽:")
        with open(positive_list_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                print(f"  {line.strip()}")

        return str(positive_list_file)

    def prepare_negative_samples(self):
        """
        準備負樣本描述檔案

        Returns:
            str: 負樣本描述檔案路徑
        """
        negative_list_file = self.data_dir / "negative_images.txt"

        # 獲取所有負樣本影像
        negative_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']:
            negative_images.extend(self.negative_dir.glob(ext))

        if not negative_images:
            raise ValueError(f"在 {self.negative_dir} 中找不到負樣本影像")

        # 寫入描述檔案，使用相對路徑
        with open(negative_list_file, 'w') as f:
            for img_path in negative_images:
                # 使用相對於專案根目錄的路徑，確保格式正確
                rel_path = str(img_path.relative_to(self.project_dir)).replace('\\', '/')
                f.write(f"{rel_path}\n")

        print(f"✓ 準備了 {len(negative_images)} 個負樣本")
        print(f"負樣本列表: {negative_list_file}")
        
        # 顯示前幾行內容以便調試
        print("負樣本列表檔案內容預覽:")
        with open(negative_list_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                print(f"  {line.strip()}")

        return str(negative_list_file)

    def create_samples_python(self, positive_list_file, negative_list_file, 
                             num_samples=None, sample_width=24, sample_height=24):
        """
        使用純Python創建訓練樣本（替代opencv_createsamples）

        Args:
            positive_list_file (str): 正樣本列表檔案
            negative_list_file (str): 負樣本列表檔案
            num_samples (int): 要生成的樣本數量
            sample_width (int): 樣本寬度
            sample_height (int): 樣本高度

        Returns:
            str: 生成的.vec檔案路徑
        """
        vec_file = self.data_dir / "positive_samples.vec"
        
        print("使用Python替代方案創建訓練樣本...")
        
        # 讀取正樣本資訊
        positive_samples = []
        with open(positive_list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:  # 檔名 + 數量 + 至少一組座標
                    img_path = self.project_dir / parts[0]
                    count = int(parts[1])
                    
                    for i in range(count):
                        if len(parts) >= 6 + i * 4:
                            x, y, w, h = map(int, parts[2 + i * 4:6 + i * 4])
                            positive_samples.append((str(img_path), x, y, w, h))

        if not positive_samples:
            raise ValueError("沒有找到有效的正樣本")

        # 限制樣本數量
        if num_samples is None:
            num_samples = min(len(positive_samples), 100)
        
        num_samples = min(num_samples, len(positive_samples))
        
        print(f"找到 {len(positive_samples)} 個正樣本，將使用 {num_samples} 個")

        # 創建vec檔案
        samples_created = 0
        vec_data = []
        
        for i, (img_path, x, y, w, h) in enumerate(positive_samples[:num_samples]):
            try:
                # 讀取影像
                img = cv2.imread(img_path)
                if img is None:
                    print(f"無法讀取影像: {img_path}")
                    continue
                
                # 提取ROI
                roi = img[y:y+h, x:x+w]
                if roi.size == 0:
                    print(f"無效的ROI: ({x}, {y}, {w}, {h}) in {img_path}")
                    continue
                
                # 調整大小
                roi_resized = cv2.resize(roi, (sample_width, sample_height))
                
                # 轉為灰階
                if len(roi_resized.shape) == 3:
                    roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                else:
                    roi_gray = roi_resized
                
                # 添加到vec資料
                vec_data.append(roi_gray)
                samples_created += 1
                
                if samples_created % 10 == 0:
                    print(f"已處理 {samples_created}/{num_samples} 個樣本")
                    
            except Exception as e:
                print(f"處理樣本時出錯 {img_path}: {e}")
                continue

        if samples_created == 0:
            raise ValueError("無法創建任何有效樣本")

        # 寫入標準OpenCV vec檔案格式
        print(f"寫入vec檔案: {vec_file}")
        
        # 計算檔案大小
        image_size = sample_width * sample_height
        
        with open(vec_file, 'wb') as f:
            # 寫入檔頭（OpenCV標準格式）
            f.write(samples_created.to_bytes(4, 'little'))     # 樣本數量
            f.write(image_size.to_bytes(4, 'little'))          # 每個樣本的大小（寬*高）
            f.write(sample_width.to_bytes(4, 'little'))        # 寬度
            f.write(sample_height.to_bytes(4, 'little'))       # 高度
            
            # 寫入每個樣本
            for i, sample in enumerate(vec_data):
                # 每個樣本前還需要寫入樣本索引（4字節）
                f.write(i.to_bytes(4, 'little'))
                
                # 確保樣本是正確的尺寸
                if sample.shape != (sample_height, sample_width):
                    sample = cv2.resize(sample, (sample_width, sample_height))
                
                # 寫入像素數據（按行優先順序）
                sample_bytes = sample.astype(np.uint8).tobytes()
                f.write(sample_bytes)

        # 驗證寫入的檔案
        file_size = vec_file.stat().st_size
        expected_size = 16 + samples_created * (4 + image_size)  # 檔頭16字節 + 每樣本(4字節索引 + 圖像數據)
        
        print(f"✓ 成功創建 {samples_created} 個訓練樣本")
        print(f"Vec檔案: {vec_file}")
        print(f"檔案大小: {file_size:,} bytes (預期: {expected_size:,} bytes)")
        
        if abs(file_size - expected_size) > 100:  # 允許小誤差
            print(f"⚠️  檔案大小不符預期，可能有格式問題")
        
        return str(vec_file)

    def create_samples_opencv_format(self, positive_list_file, negative_list_file, 
                                   num_samples=None, sample_width=24, sample_height=24):
        """
        創建符合OpenCV格式的vec檔案的備選方法
        """
        print("嘗試使用opencv-python直接創建vec檔案...")
        
        vec_file = self.data_dir / "positive_samples_alt.vec"
        
        # 讀取所有正樣本
        samples = []
        with open(positive_list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    img_path = self.project_dir / parts[0]
                    count = int(parts[1])
                    
                    # 讀取影像
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    for i in range(count):
                        if len(parts) >= 6 + i * 4:
                            x, y, w, h = map(int, parts[2 + i * 4:6 + i * 4])
                            
                            # 提取ROI並調整尺寸
                            roi = img[y:y+h, x:x+w]
                            if roi.size > 0:
                                resized = cv2.resize(roi, (sample_width, sample_height))
                                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
                                samples.append(gray)

        if not samples:
            raise ValueError("無法創建任何樣本")

        # 限制數量
        if num_samples and num_samples < len(samples):
            samples = samples[:num_samples]

        print(f"準備寫入 {len(samples)} 個樣本到vec檔案")

        # 使用更簡化的格式
        with open(vec_file, 'wb') as f:
            # 簡化的檔頭
            f.write(len(samples).to_bytes(4, 'little'))
            f.write((sample_width * sample_height).to_bytes(4, 'little'))
            f.write(sample_width.to_bytes(4, 'little'))
            f.write(sample_height.to_bytes(4, 'little'))
            
            # 直接寫入圖像數據，不使用額外的索引
            for sample in samples:
                f.write(sample.tobytes())

        print(f"✓ 創建替代vec檔案: {vec_file}")
        return str(vec_file)

    def create_samples(self, positive_list_file, negative_list_file, 
                      num_samples=None, sample_width=24, sample_height=24):
        """
        創建訓練樣本（嘗試多種方法）

        Args:
            positive_list_file (str): 正樣本列表檔案
            negative_list_file (str): 負樣本列表檔案
            num_samples (int): 要生成的樣本數量（如果為None則自動計算）
            sample_width (int): 樣本寬度
            sample_height (int): 樣本高度

        Returns:
            str: 生成的.vec檔案路徑
        """
        # 計算實際可用的標註數量
        total_annotations = 0
        with open(positive_list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        annotation_count = int(parts[1])
                        total_annotations += annotation_count
                    except ValueError:
                        continue

        # 如果沒有指定樣本數量，則根據可用的標註數量計算
        if num_samples is None:
            # 使用可用標註的80%，但不超過100（適合小資料集）
            num_samples = min(int(total_annotations * 0.8), 100)
            print(f"自動設定樣本數量為: {num_samples}（基於 {total_annotations} 個標註）")

        # 首先嘗試使用opencv_createsamples（但我們知道會失敗）
        print("嘗試使用 opencv_createsamples（預期會失敗）...")
        
        # 直接使用Python方法，因為我們知道opencv_createsamples有問題
        try:
            # 先嘗試標準格式
            return self.create_samples_python(
                positive_list_file, negative_list_file,
                num_samples, sample_width, sample_height
            )
        except Exception as e:
            print(f"標準格式失敗: {e}")
            print("嘗試替代格式...")
            
            # 如果標準格式失敗，嘗試簡化格式
            return self.create_samples_opencv_format(
                positive_list_file, negative_list_file,
                num_samples, sample_width, sample_height
            )

    def train_cascade_python_check(self):
        """
        檢查是否可以使用opencv_traincascade
        """
        try:
            result = subprocess.run(["opencv_traincascade"], 
                                  capture_output=True, text=True, timeout=10)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def safe_rmtree(self, path):
        """
        安全地刪除目錄，處理權限問題
        
        Args:
            path: 要刪除的目錄路徑
        """
        if not path.exists():
            return
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                print(f"嘗試清理目錄 {path} (第{attempt+1}次)")
                
                # 方法1: 嘗試修改所有檔案權限
                for root, dirs, files in os.walk(path):
                    for d in dirs:
                        dir_path = os.path.join(root, d)
                        try:
                            os.chmod(dir_path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
                        except:
                            pass
                    for f in files:
                        file_path = os.path.join(root, f)
                        try:
                            os.chmod(file_path, stat.S_IWRITE | stat.S_IREAD)
                        except:
                            pass
                
                # 嘗試刪除
                def handle_remove_readonly(func, path, exc):
                    """處理只讀檔案的刪除"""
                    try:
                        if os.path.exists(path):
                            os.chmod(path, stat.S_IWRITE)
                            func(path)
                    except:
                        pass
                
                shutil.rmtree(path, onerror=handle_remove_readonly)
                print(f"✓ 成功清理目錄")
                return
                
            except Exception as e:
                print(f"清理嘗試 {attempt+1} 失敗: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(1)  # 等待一秒後重試
                continue
        
        # 如果所有嘗試都失敗，創建新目錄
        print(f"⚠️  無法清理目錄 {path}，創建新的輸出目錄...")
        import uuid
        timestamp = int(time.time())
        new_name = f"trained_model_{timestamp}"
        self.output_dir = self.data_dir / new_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"使用新目錄: {self.output_dir}")

    def train_cascade(self, vec_file, negative_list_file, 
                     num_pos=None, num_neg=None, num_stages=20,
                     sample_width=24, sample_height=24):
        """
        訓練Haar cascade分類器

        Args:
            vec_file (str): 正樣本.vec檔案
            negative_list_file (str): 負樣本列表檔案
            num_pos (int): 正樣本數量（如果為None則自動計算）
            num_neg (int): 負樣本數量（如果為None則自動計算）
            num_stages (int): 訓練階段數
            sample_width (int): 樣本寬度
            sample_height (int): 樣本高度
        """
        # 檢查opencv_traincascade是否可用
        if not self.train_cascade_python_check():
            print("✗ 找不到 opencv_traincascade 工具")
            print("請安裝完整的OpenCV套件:")
            print("  conda install opencv")
            print("  或下載OpenCV官方版本")
            
            # 嘗試提供替代建議
            print("\n替代方案:")
            print("1. 使用線上Colab環境:")
            print("   https://colab.research.google.com/")
            print("2. 使用Docker容器:")
            print("   docker run -it opencv/opencv:latest")
            print("3. 安裝完整的OpenCV-contrib-python:")
            print("   pip uninstall opencv-python")
            print("   pip install opencv-contrib-python")
            
            return None

        # 自動計算合適的正負樣本數量（更保守的設定以減少誤偵測）
        if num_pos is None:
            # 從正樣本列表檔案計算實際樣本數
            try:
                with open(self.data_dir / "positive_images.txt", 'r') as f:
                    lines = f.readlines()
                total_annotations = 0
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        total_annotations += int(parts[1])
                
                # 使用20%的標註作為訓練樣本（更保守，減少過擬合）
                num_pos = min(int(total_annotations * 0.2), 20)
            except:
                num_pos = 15  # 非常保守的預設值
            print(f"自動設定正樣本數量為: {num_pos}")

        if num_neg is None:
            with open(negative_list_file, 'r') as f:
                total_neg = len(f.readlines())
            # 增加負樣本比例以減少誤偵測
            num_neg = min(int(total_neg * 0.4), num_pos * 5)  # 負樣本為正樣本的5倍
            print(f"自動設定負樣本數量為: {num_neg}")

        # 安全地清理輸出目錄
        print(f"準備輸出目錄: {self.output_dir}")
        self.safe_rmtree(self.output_dir)
        
        # 確保目錄存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 驗證目錄是否可寫
        test_file = self.output_dir / "test_write.txt"
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            test_file.unlink()
            print(f"✓ 輸出目錄可寫入")
        except Exception as e:
            print(f"✗ 輸出目錄無法寫入: {e}")
            # 最後嘗試：使用臨時目錄
            import tempfile
            temp_dir = Path(tempfile.mkdtemp(prefix="haar_training_"))
            print(f"使用臨時目錄: {temp_dir}")
            self.output_dir = temp_dir

        # 構建opencv_traincascade命令 - 調整參數以減少誤偵測
        cmd = [
            "opencv_traincascade",
            "-data", str(self.output_dir),
            "-vec", vec_file,
            "-bg", negative_list_file,
            "-numPos", str(num_pos),
            "-numNeg", str(num_neg),
            "-numStages", str(num_stages),
            "-w", str(sample_width),
            "-h", str(sample_height),
            "-minHitRate", "0.999",        # 提高命中率要求（減少漏偵測）
            "-maxFalseAlarmRate", "0.3",   # 降低誤報率容忍度（減少誤偵測）
            "-weightTrimRate", "0.95",
            "-maxDepth", "2",              # 增加決策樹深度（提高精確度）
            "-maxWeakCount", "150",        # 增加弱分類器數量（提高精確度）
            "-mode", "ALL",                # 使用所有特徵類型
            "-precalcValBufSize", "2048",  # 增加緩衝區大小
            "-precalcIdxBufSize", "2048"
        ]

        print(f"開始訓練高精確度Haar cascade分類器...")
        print(f"⚠️  使用更嚴格的參數以減少誤偵測")
        print(f"預計需要時間: 30分鐘到4小時（取決於資料量）")
        print(f"訓練參數:")
        print(f"  正樣本: {num_pos}")
        print(f"  負樣本: {num_neg}")
        print(f"  階段數: {num_stages}")
        print(f"  樣本尺寸: {sample_width}x{sample_height}")
        print(f"  命中率要求: 99.9%")
        print(f"  誤報率容忍: 30%")
        print(f"  輸出目錄: {self.output_dir}")
        print(f"\n⏱️  訓練開始時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("按 Ctrl+C 可中斷訓練")

        try:
            # 使用Popen以便即時顯示輸出
            start_time = time.time()
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT, text=True,
                                     cwd=str(self.project_dir))

            # 即時顯示訓練進度
            stage_count = 0
            for line in process.stdout:
                line_clean = line.rstrip()
                print(line_clean)
                
                # 追蹤訓練進度
                if "STAGE" in line_clean:
                    stage_count += 1
                    progress = (stage_count / num_stages) * 100
                    elapsed = time.time() - start_time
                    estimated_total = elapsed / stage_count * num_stages
                    remaining = estimated_total - elapsed
                    
                    print(f">>> 訓練進度: 第 {stage_count}/{num_stages} 階段 ({progress:.1f}%)")
                    print(f"    已用時間: {elapsed/60:.1f} 分鐘")
                    if remaining > 0:
                        print(f"    預計剩餘: {remaining/60:.1f} 分鐘")

            process.wait()

            if process.returncode == 0:
                cascade_file = self.output_dir / "cascade.xml"
                if cascade_file.exists():
                    total_time = time.time() - start_time
                    print(f"\n🎉 高精確度模型訓練完成!")
                    print(f"⏱️  總訓練時間: {total_time/60:.1f} 分鐘")
                    print(f"📁 模型檔案: {cascade_file}")
                    
                    # 顯示模型資訊
                    file_size = cascade_file.stat().st_size
                    print(f"📊 模型檔案大小: {file_size:,} bytes")
                    
                    # 如果使用臨時目錄，複製到原目標位置
                    if "temp" in str(self.output_dir):
                        try:
                            final_dir = self.data_dir / "trained_model"
                            final_dir.mkdir(exist_ok=True)
                            final_cascade = final_dir / "cascade.xml"
                            shutil.copy2(cascade_file, final_cascade)
                            print(f"📋 模型已複製到: {final_cascade}")
                            cascade_file = final_cascade
                        except Exception as e:
                            print(f"⚠️  無法複製模型檔案: {e}")
                    
                    # 提供使用建議
                    print("\n📋 高精確度模型使用說明:")
                    print(f"1. 測試模型: python demo.py --cascade {cascade_file}")
                    print("2. 建議的偵測參數（減少誤偵測）:")
                    print("   - scaleFactor: 1.1-1.2（較大值，減少計算）")
                    print("   - minNeighbors: 5-10（較大值，減少誤偵測）")
                    print("   - minSize: (40,40) 或更大（過濾小物件）")
                    print("   - maxSize: 設定合理的最大尺寸")
                    print("\n3. 如果仍有誤偵測:")
                    print("   - 收集更多負樣本（包含常見的誤偵測物件）")
                    print("   - 增加訓練階段數到15-20")
                    print("   - 檢查並改善正樣本標註品質")
                    
                    return str(cascade_file)
                else:
                    raise RuntimeError("訓練完成但找不到cascade.xml檔案")
            else:
                raise RuntimeError(f"訓練失敗，退出碼: {process.returncode}")

        except KeyboardInterrupt:
            print(f"\n⏹️  訓練被用戶中斷")
            if process:
                process.terminate()
                print("正在清理...")
                time.sleep(2)
            raise
        except FileNotFoundError:
            print("✗ 找不到 opencv_traincascade 工具")
            print("請確保已安裝完整的OpenCV套件")
            raise

    def validate_data(self):
        """
        驗證訓練資料的完整性
        """
        print("驗證訓練資料...")

        # 檢查正樣本（避免重複計算同名檔案，不分大小寫）
        positive_images = []
        seen_pos = set()
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']:
            for img in self.positive_dir.glob(ext):
                name_lower = img.name.lower()
                if name_lower not in seen_pos:
                    positive_images.append(img)
                    seen_pos.add(name_lower)
        print(f"正樣本影像: {len(positive_images)}")

        # 檢查負樣本（避免重複計算同名檔案，不分大小寫）
        negative_images = []
        seen_neg = set()
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']:
            for img in self.negative_dir.glob(ext):
                name_lower = img.name.lower()
                if name_lower not in seen_neg:
                    negative_images.append(img)
                    seen_neg.add(name_lower)
        print(f"負樣本影像: {len(negative_images)}")

        # 檢查標註檔案
        annotation_files = list(self.annotations_dir.glob("*.txt"))
        print(f"標註檔案: {len(annotation_files)}")

        # 提供建議
        issues = []
        if len(positive_images) < 50:
            issues.append(f"正樣本影像太少（當前: {len(positive_images)}，建議至少50張）")

        if len(negative_images) < 100:
            issues.append(f"負樣本影像太少（當前: {len(negative_images)}，建議至少100張）")

        if len(annotation_files) < len(positive_images) * 0.8:
            issues.append(f"標註檔案過少（當前: {len(annotation_files)}，應接近正樣本數量）")

        if issues:
            print("⚠️  發現以下問題:")
            for issue in issues:
                print(f"   {issue}")
            return False

        print("✓ 資料驗證通過")
        return True

def main():
    parser = argparse.ArgumentParser(description='Haar Cascade Training for Scooter Detection')
    parser.add_argument('--project_dir', default='./', help='專案目錄路徑')
    parser.add_argument('--num_samples', type=int, help='生成樣本數量（自動計算如果未指定）')
    parser.add_argument('--num_pos', type=int, help='訓練正樣本數量（自動計算如果未指定）')
    parser.add_argument('--num_neg', type=int, help='訓練負樣本數量（自動計算如果未指定）')
    parser.add_argument('--num_stages', type=int, default=12, help='訓練階段數（預設12，平衡精度和速度）')
    parser.add_argument('--width', type=int, default=24, help='樣本寬度')
    parser.add_argument('--height', type=int, default=24, help='樣本高度')
    parser.add_argument('--validate_only', action='store_true', help='僅驗證資料')
    parser.add_argument('--python_only', action='store_true', help='強制使用Python替代方案')
    parser.add_argument('--force_clean', action='store_true', help='強制清理輸出目錄')
    parser.add_argument('--high_precision', action='store_true', help='高精確度模式（減少誤偵測）')

    args = parser.parse_args()

    try:
        # 創建訓練器
        trainer = HaarCascadeTrainer(args.project_dir)

        # 高精確度模式調整
        if args.high_precision:
            print("🎯 啟用高精確度模式")
            args.num_stages = max(args.num_stages, 15)  # 至少15個階段
            print(f"調整訓練階段數為: {args.num_stages}")

        # 強制清理（如果指定）
        if args.force_clean:
            print("強制清理輸出目錄...")
            trainer.safe_rmtree(trainer.output_dir)

        # 驗證資料
        if not trainer.validate_data():
            if not args.validate_only:
                print("✗ 資料驗證失敗，但仍嘗試訓練...")
                print("\n💡 減少誤偵測的改善建議:")
                print("1. 收集更多多樣化的負樣本（包含容易誤偵測的物件）")
                print("2. 確保正樣本標註精確（緊貼滑板車邊界）")
                print("3. 增加不同角度、光照、距離的正樣本")
                print("4. 使用 --high_precision 模式訓練")
                print("5. 考慮增加訓練階段數到15-20")
            else:
                return 1

        if args.validate_only:
            print("✓ 資料驗證完成")
            return 0

        # 準備樣本
        print("\n=== 準備負樣本 ===")
        negative_list = trainer.prepare_negative_samples()

        print("\n=== 準備正樣本 ===")
        positive_list = trainer.prepare_positive_samples()

        print("\n=== 創建訓練樣本 ===")
        if args.python_only:
            vec_file = trainer.create_samples_python(
                positive_list, negative_list,
                num_samples=args.num_samples,
                sample_width=args.width,
                sample_height=args.height
            )
        else:
            vec_file = trainer.create_samples(
                positive_list, negative_list,
                num_samples=args.num_samples,
                sample_width=args.width,
                sample_height=args.height
            )

        print("\n=== 開始訓練 ===")
        if args.high_precision:
            print("🎯 使用高精確度模式訓練...")
            
        cascade_file = trainer.train_cascade(
            vec_file, negative_list,
            num_pos=args.num_pos,
            num_neg=args.num_neg,
            num_stages=args.num_stages,
            sample_width=args.width,
            sample_height=args.height
        )

        if cascade_file:
            print(f"\n🎉 訓練完成! 模型檔案: {cascade_file}")
            print("\n📝 測試和調優指南:")
            print("1. 基本測試:")
            print(f"   python demo.py --cascade {cascade_file} --webcam")
            print("\n2. 如果有誤偵測，調整偵測參數:")
            print("   - 增加 minNeighbors (5-10)")
            print("   - 增加 scaleFactor (1.1-1.2)")
            print("   - 設定 minSize 過濾小物件")
            print("   - 設定 maxSize 限制大物件")
            print("\n3. 如果效果仍不佳:")
            print("   - 收集更多包含誤偵測物件的負樣本")
            print("   - 使用 --high_precision --num_stages 20 重新訓練")
            print("   - 檢查並改善標註品質")
        else:
            print("\n❌ 訓練失敗，請檢查錯誤訊息並重試")

    except KeyboardInterrupt:
        print("\n🛑 程序被用戶中斷")
        return 0
    except Exception as e:
        print(f"✗ 訓練失敗: {e}")
        import traceback
        print("\n詳細錯誤資訊:")
        traceback.print_exc()
        
        print("\n🔧 常見問題解決方案:")
        print("1. 權限問題: 以管理員身份執行或更改專案目錄")
        print("2. OpenCV工具缺失: 安裝 opencv-contrib-python")
        print("3. 記憶體不足: 減少樣本數量或階段數")
        print("4. 資料問題: 檢查標註檔案格式和影像完整性")
        
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
