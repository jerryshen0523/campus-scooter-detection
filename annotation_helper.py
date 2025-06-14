#!/usr/bin/env python3
"""
Annotation Helper Utilities
標註輔助工具模組

提供標註檔案轉換、驗證和處理功能
"""

import cv2
import numpy as np
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import os

class AnnotationConverter:
    """標註格式轉換器"""

    @staticmethod
    def yolo_to_pascal_voc(yolo_annotation, image_width, image_height, class_names=None):
        """
        將YOLO格式轉換為Pascal VOC格式

        Args:
            yolo_annotation: YOLO格式標註 (class, x_center, y_center, width, height)
            image_width: 影像寬度
            image_height: 影像高度
            class_names: 類別名稱列表

        Returns:
            Pascal VOC格式字典
        """
        if class_names is None:
            class_names = ['scooter']

        class_id, x_center, y_center, width, height = yolo_annotation

        # 轉換為絕對座標
        x_center *= image_width
        y_center *= image_height
        width *= image_width
        height *= image_height

        # 計算邊界框座標
        xmin = int(x_center - width / 2)
        ymin = int(y_center - height / 2)
        xmax = int(x_center + width / 2)
        ymax = int(y_center + height / 2)

        # 確保座標在影像範圍內
        xmin = max(0, min(xmin, image_width - 1))
        ymin = max(0, min(ymin, image_height - 1))
        xmax = max(0, min(xmax, image_width - 1))
        ymax = max(0, min(ymax, image_height - 1))

        return {
            'name': class_names[int(class_id)],
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        }

    @staticmethod
    def pascal_voc_to_yolo(pascal_annotation, image_width, image_height):
        """
        將Pascal VOC格式轉換為YOLO格式

        Args:
            pascal_annotation: Pascal VOC格式標註字典
            image_width: 影像寬度
            image_height: 影像高度

        Returns:
            YOLO格式元組 (class_id, x_center, y_center, width, height)
        """
        xmin = pascal_annotation['xmin']
        ymin = pascal_annotation['ymin']
        xmax = pascal_annotation['xmax']
        ymax = pascal_annotation['ymax']

        # 計算中心點和尺寸
        width = xmax - xmin
        height = ymax - ymin
        x_center = xmin + width / 2
        y_center = ymin + height / 2

        # 正規化為相對座標
        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height

        return (0, x_center, y_center, width, height)  # 假設scooter的class_id為0

    @staticmethod
    def create_pascal_voc_xml(image_path, annotations, output_path):
        """
        創建Pascal VOC格式的XML標註檔案

        Args:
            image_path: 影像檔案路徑
            annotations: 標註列表
            output_path: 輸出XML檔案路徑
        """
        # 讀取影像資訊
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"無法讀取影像: {image_path}")

        height, width, depth = image.shape

        # 創建XML結構
        annotation = ET.Element('annotation')

        # 檔案資訊
        folder = ET.SubElement(annotation, 'folder')
        folder.text = str(Path(image_path).parent.name)

        filename = ET.SubElement(annotation, 'filename')
        filename.text = str(Path(image_path).name)

        path = ET.SubElement(annotation, 'path')
        path.text = str(image_path)

        # 來源資訊
        source = ET.SubElement(annotation, 'source')
        database = ET.SubElement(source, 'database')
        database.text = 'Campus Scooter Dataset'

        # 影像尺寸
        size = ET.SubElement(annotation, 'size')
        width_elem = ET.SubElement(size, 'width')
        width_elem.text = str(width)
        height_elem = ET.SubElement(size, 'height')
        height_elem.text = str(height)
        depth_elem = ET.SubElement(size, 'depth')
        depth_elem.text = str(depth)

        segmented = ET.SubElement(annotation, 'segmented')
        segmented.text = '0'

        # 物件標註
        for ann in annotations:
            obj = ET.SubElement(annotation, 'object')

            name = ET.SubElement(obj, 'name')
            name.text = ann['name']

            pose = ET.SubElement(obj, 'pose')
            pose.text = 'Unspecified'

            truncated = ET.SubElement(obj, 'truncated')
            truncated.text = '0'

            difficult = ET.SubElement(obj, 'difficult')
            difficult.text = '0'

            bndbox = ET.SubElement(obj, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(ann['xmin'])
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(ann['ymin'])
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(ann['xmax'])
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(ann['ymax'])

        # 寫入XML檔案
        tree = ET.ElementTree(annotation)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)

class AnnotationValidator:
    """標註驗證器"""

    @staticmethod
    def validate_yolo_annotation(annotation_line, image_path=None):
        """
        驗證YOLO格式標註

        Args:
            annotation_line: 標註行文字
            image_path: 影像檔案路徑（可選，用於更詳細驗證）

        Returns:
            dict: 驗證結果
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        try:
            parts = annotation_line.strip().split()

            if len(parts) < 5:
                result['valid'] = False
                result['errors'].append(f"格式錯誤：需要至少5個數值，got {len(parts)}")
                return result

            # 檢查數值格式
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError as e:
                result['valid'] = False
                result['errors'].append(f"數值格式錯誤: {e}")
                return result

            # 檢查座標範圍
            if not (0 <= x_center <= 1):
                result['errors'].append(f"x_center超出範圍 [0,1]: {x_center}")
            if not (0 <= y_center <= 1):
                result['errors'].append(f"y_center超出範圍 [0,1]: {y_center}")
            if not (0 < width <= 1):
                result['errors'].append(f"width超出範圍 (0,1]: {width}")
            if not (0 < height <= 1):
                result['errors'].append(f"height超出範圍 (0,1]: {height}")

            # 檢查邊界框是否超出影像範圍
            xmin = x_center - width / 2
            ymin = y_center - height / 2
            xmax = x_center + width / 2
            ymax = y_center + height / 2

            if xmin < 0 or ymin < 0 or xmax > 1 or ymax > 1:
                result['warnings'].append("邊界框部分超出影像範圍")

            # 檢查邊界框大小
            if width < 0.01 or height < 0.01:
                result['warnings'].append("邊界框過小，可能影響訓練效果")
            if width > 0.8 or height > 0.8:
                result['warnings'].append("邊界框過大，可能包含過多背景")

            if result['errors']:
                result['valid'] = False

        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"未知錯誤: {e}")

        return result

    @staticmethod
    def validate_dataset_consistency(image_dir, annotation_dir):
        """
        驗證資料集一致性

        Args:
            image_dir: 影像目錄
            annotation_dir: 標註目錄

        Returns:
            dict: 驗證報告
        """
        image_path = Path(image_dir)
        annotation_path = Path(annotation_dir)

        report = {
            'total_images': 0,
            'total_annotations': 0,
            'matched_pairs': 0,
            'missing_annotations': [],
            'missing_images': [],
            'invalid_annotations': [],
            'valid': True
        }

        # 獲取所有影像檔案
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = {}
        for ext in image_extensions:
            for img_file in image_path.glob(f'*{ext}'):
                image_files[img_file.stem] = img_file

        report['total_images'] = len(image_files)

        # 獲取所有標註檔案
        annotation_files = {}
        for ann_file in annotation_path.glob('*.txt'):
            annotation_files[ann_file.stem] = ann_file

        report['total_annotations'] = len(annotation_files)

        # 檢查匹配對
        for img_stem, img_file in image_files.items():
            if img_stem in annotation_files:
                report['matched_pairs'] += 1

                # 驗證標註檔案
                ann_file = annotation_files[img_stem]
                try:
                    with open(ann_file, 'r') as f:
                        lines = f.readlines()

                    valid_lines = 0
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            validation = AnnotationValidator.validate_yolo_annotation(line)
                            if validation['valid']:
                                valid_lines += 1
                            else:
                                report['invalid_annotations'].append({
                                    'file': str(ann_file),
                                    'line': line,
                                    'errors': validation['errors']
                                })

                    if valid_lines == 0:
                        report['invalid_annotations'].append({
                            'file': str(ann_file),
                            'error': '沒有有效的標註行'
                        })

                except Exception as e:
                    report['invalid_annotations'].append({
                        'file': str(ann_file),
                        'error': f'檔案讀取錯誤: {e}'
                    })
            else:
                report['missing_annotations'].append(str(img_file))

        # 檢查孤兒標註檔案
        for ann_stem, ann_file in annotation_files.items():
            if ann_stem not in image_files:
                report['missing_images'].append(str(ann_file))

        # 判斷整體有效性
        if (report['missing_annotations'] or 
            report['missing_images'] or 
            report['invalid_annotations']):
            report['valid'] = False

        return report

def convert_dataset_format(input_dir, output_dir, input_format='yolo', output_format='pascal_voc'):
    """
    轉換資料集格式

    Args:
        input_dir: 輸入目錄
        output_dir: 輸出目錄
        input_format: 輸入格式 ('yolo' 或 'pascal_voc')
        output_format: 輸出格式 ('yolo' 或 'pascal_voc')
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if input_format == 'yolo' and output_format == 'pascal_voc':
        # YOLO轉Pascal VOC
        annotation_dir = input_path / 'annotations'
        image_dir = input_path / 'images'

        for ann_file in annotation_dir.glob('*.txt'):
            image_file = None
            for ext in ['.jpg', '.jpeg', '.png']:
                potential_image = image_dir / f"{ann_file.stem}{ext}"
                if potential_image.exists():
                    image_file = potential_image
                    break

            if image_file is None:
                print(f"找不到對應的影像檔案: {ann_file.stem}")
                continue

            # 讀取影像尺寸
            image = cv2.imread(str(image_file))
            if image is None:
                continue

            height, width = image.shape[:2]

            # 讀取YOLO標註
            annotations = []
            with open(ann_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 5:
                            yolo_ann = [float(x) for x in parts[:5]]
                            pascal_ann = AnnotationConverter.yolo_to_pascal_voc(
                                yolo_ann, width, height
                            )
                            annotations.append(pascal_ann)

            # 創建Pascal VOC XML
            if annotations:
                xml_output = output_path / f"{ann_file.stem}.xml"
                AnnotationConverter.create_pascal_voc_xml(
                    image_file, annotations, xml_output
                )
                print(f"轉換: {ann_file.name} -> {xml_output.name}")

def print_validation_report(report):
    """
    列印驗證報告

    Args:
        report: 驗證報告字典
    """
    print("=== 資料集驗證報告 ===")
    print(f"總影像數: {report['total_images']}")
    print(f"總標註數: {report['total_annotations']}")
    print(f"匹配對數: {report['matched_pairs']}")
    print(f"整體狀態: {'✓ 有效' if report['valid'] else '✗ 有問題'}")

    if report['missing_annotations']:
        print(f"\n缺少標註的影像 ({len(report['missing_annotations'])}):")
        for img in report['missing_annotations'][:5]:
            print(f"  - {Path(img).name}")
        if len(report['missing_annotations']) > 5:
            print(f"  ... 還有 {len(report['missing_annotations'])-5} 個檔案")

    if report['missing_images']:
        print(f"\n缺少影像的標註 ({len(report['missing_images'])}):")
        for ann in report['missing_images'][:5]:
            print(f"  - {Path(ann).name}")
        if len(report['missing_images']) > 5:
            print(f"  ... 還有 {len(report['missing_images'])-5} 個檔案")

    if report['invalid_annotations']:
        print(f"\n無效標註 ({len(report['invalid_annotations'])}):")
        for invalid in report['invalid_annotations'][:3]:
            print(f"  - {Path(invalid['file']).name}")
            if 'errors' in invalid:
                for error in invalid['errors']:
                    print(f"    錯誤: {error}")
        if len(report['invalid_annotations']) > 3:
            print(f"  ... 還有 {len(report['invalid_annotations'])-3} 個檔案")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Annotation Helper Utilities')
    parser.add_argument('--action', choices=['validate', 'convert'], 
                       required=True, help='執行動作')
    parser.add_argument('--input', required=True, help='輸入目錄')
    parser.add_argument('--output', help='輸出目錄')
    parser.add_argument('--input_format', choices=['yolo', 'pascal_voc'], 
                       default='yolo', help='輸入格式')
    parser.add_argument('--output_format', choices=['yolo', 'pascal_voc'], 
                       default='pascal_voc', help='輸出格式')

    args = parser.parse_args()

    if args.action == 'validate':
        # 假設輸入目錄包含images和annotations子目錄
        image_dir = Path(args.input) / 'positive'
        annotation_dir = Path(args.input) / 'annotations'

        if not image_dir.exists() or not annotation_dir.exists():
            print("輸入目錄應包含 'positive' 和 'annotations' 子目錄")
            exit(1)

        report = AnnotationValidator.validate_dataset_consistency(
            image_dir, annotation_dir
        )
        print_validation_report(report)

    elif args.action == 'convert':
        if not args.output:
            print("convert動作需要指定 --output 參數")
            exit(1)

        convert_dataset_format(
            args.input, args.output, 
            args.input_format, args.output_format
        )
