#!/usr/bin/env python3
"""
Image Processing Utilities
影像處理工具模組

提供影像預處理、增強和分析功能
"""

import cv2
import numpy as np
from pathlib import Path
import os

class ImageProcessor:
    """影像處理器類別"""

    @staticmethod
    def enhance_contrast(image, alpha=1.5, beta=0):
        """
        增強影像對比度

        Args:
            image: 輸入影像
            alpha: 對比度調整因子
            beta: 亮度調整因子

        Returns:
            增強後的影像
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    @staticmethod
    def adjust_brightness(image, value=30):
        """
        調整影像亮度

        Args:
            image: 輸入影像
            value: 亮度調整值 (-100 到 100)

        Returns:
            調整後的影像
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        v = cv2.add(v, value)
        v = np.clip(v, 0, 255)

        hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def denoise_image(image, h=10, template_window_size=7, search_window_size=21):
        """
        影像去噪

        Args:
            image: 輸入影像
            h: 濾波強度
            template_window_size: 模板視窗大小
            search_window_size: 搜尋視窗大小

        Returns:
            去噪後的影像
        """
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, 
                                               template_window_size, search_window_size)

    @staticmethod
    def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
        """
        保持長寬比的影像縮放

        Args:
            image: 輸入影像
            width: 目標寬度
            height: 目標高度
            inter: 插值方法

        Returns:
            縮放後的影像
        """
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=inter)

    @staticmethod
    def create_augmented_dataset(image_path, output_dir, num_augments=5):
        """
        創建增強資料集

        Args:
            image_path: 原始影像路徑
            output_dir: 輸出目錄
            num_augments: 增強數量
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        base_name = Path(image_path).stem

        # 原始影像
        cv2.imwrite(str(output_path / f"{base_name}_original.jpg"), image)

        # 生成增強影像
        for i in range(num_augments):
            augmented = image.copy()

            # 隨機調整亮度和對比度
            alpha = np.random.uniform(0.8, 1.2)  # 對比度
            beta = np.random.randint(-30, 30)    # 亮度
            augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)

            # 隨機噪聲
            if np.random.random() > 0.5:
                noise = np.random.randint(0, 25, augmented.shape, dtype=np.uint8)
                augmented = cv2.add(augmented, noise)

            # 隨機模糊
            if np.random.random() > 0.7:
                ksize = np.random.choice([3, 5])
                augmented = cv2.GaussianBlur(augmented, (ksize, ksize), 0)

            # 隨機翻轉
            if np.random.random() > 0.5:
                augmented = cv2.flip(augmented, 1)  # 水平翻轉

            # 保存增強影像
            filename = f"{base_name}_aug_{i:02d}.jpg"
            cv2.imwrite(str(output_path / filename), augmented)

        print(f"✓ 為 {image_path} 生成了 {num_augments + 1} 張影像")

def batch_resize_images(input_dir, output_dir, target_size=(640, 480)):
    """
    批次調整影像大小

    Args:
        input_dir: 輸入目錄
        output_dir: 輸出目錄
        target_size: 目標尺寸 (width, height)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    processed_count = 0

    for image_file in input_path.iterdir():
        if image_file.suffix.lower() in image_extensions:
            # 讀取影像
            image = cv2.imread(str(image_file))
            if image is None:
                continue

            # 調整大小
            resized = cv2.resize(image, target_size)

            # 保存
            output_file = output_path / image_file.name
            cv2.imwrite(str(output_file), resized)
            processed_count += 1

    print(f"✓ 處理了 {processed_count} 張影像")

def analyze_dataset_statistics(dataset_dir):
    """
    分析資料集統計資訊

    Args:
        dataset_dir: 資料集目錄

    Returns:
        dict: 統計資訊
    """
    dataset_path = Path(dataset_dir)

    stats = {
        'total_images': 0,
        'size_distribution': {},
        'format_distribution': {},
        'average_size': [0, 0],
        'min_size': [float('inf'), float('inf')],
        'max_size': [0, 0]
    }

    total_width, total_height = 0, 0
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    for image_file in dataset_path.rglob('*'):
        if image_file.suffix.lower() in image_extensions:
            # 讀取影像資訊
            image = cv2.imread(str(image_file))
            if image is None:
                continue

            h, w = image.shape[:2]
            ext = image_file.suffix.lower()

            # 統計
            stats['total_images'] += 1
            stats['format_distribution'][ext] = stats['format_distribution'].get(ext, 0) + 1

            size_key = f"{w}x{h}"
            stats['size_distribution'][size_key] = stats['size_distribution'].get(size_key, 0) + 1

            total_width += w
            total_height += h

            # 更新最小最大尺寸
            if w < stats['min_size'][0]:
                stats['min_size'][0] = w
            if h < stats['min_size'][1]:
                stats['min_size'][1] = h
            if w > stats['max_size'][0]:
                stats['max_size'][0] = w
            if h > stats['max_size'][1]:
                stats['max_size'][1] = h

    # 計算平均尺寸
    if stats['total_images'] > 0:
        stats['average_size'] = [
            total_width // stats['total_images'],
            total_height // stats['total_images']
        ]

    return stats

def print_dataset_stats(stats):
    """
    列印資料集統計資訊

    Args:
        stats: 統計資訊字典
    """
    print("=== 資料集統計資訊 ===")
    print(f"總影像數: {stats['total_images']}")
    print(f"平均尺寸: {stats['average_size'][0]} x {stats['average_size'][1]}")
    print(f"最小尺寸: {stats['min_size'][0]} x {stats['min_size'][1]}")
    print(f"最大尺寸: {stats['max_size'][0]} x {stats['max_size'][1]}")

    print("\n格式分布:")
    for fmt, count in stats['format_distribution'].items():
        print(f"  {fmt}: {count}")

    print("\n尺寸分布 (前10個):")
    sorted_sizes = sorted(stats['size_distribution'].items(), 
                         key=lambda x: x[1], reverse=True)
    for size, count in sorted_sizes[:10]:
        print(f"  {size}: {count}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Image Processing Utilities')
    parser.add_argument('--action', choices=['resize', 'augment', 'stats'], 
                       required=True, help='處理動作')
    parser.add_argument('--input', required=True, help='輸入目錄或檔案')
    parser.add_argument('--output', help='輸出目錄')
    parser.add_argument('--size', default='640,480', help='目標尺寸 (width,height)')
    parser.add_argument('--augments', type=int, default=5, help='增強數量')

    args = parser.parse_args()

    if args.action == 'resize':
        if not args.output:
            print("resize動作需要指定 --output 參數")
            exit(1)

        width, height = map(int, args.size.split(','))
        batch_resize_images(args.input, args.output, (width, height))

    elif args.action == 'augment':
        if not args.output:
            print("augment動作需要指定 --output 參數")
            exit(1)

        if os.path.isfile(args.input):
            ImageProcessor.create_augmented_dataset(args.input, args.output, args.augments)
        else:
            for image_file in Path(args.input).glob('*.jpg'):
                ImageProcessor.create_augmented_dataset(image_file, args.output, args.augments)

    elif args.action == 'stats':
        stats = analyze_dataset_statistics(args.input)
        print_dataset_stats(stats)
