校園滑板車偵測系統 Campus Scooter Detection System
Python OpenCV License

基於Haar級聯分類器的校園滑板車即時偵測系統
Real-time campus scooter detection system using Haar cascade classifiers

📖 專案簡介 Project Overview
使用攝影機即時偵測校園內的滑板車，並提供完整的資料收集、標註、訓練和偵測流程。

🎯 專題目標
收集並標註元智大學校園滑板車影像資料集
訓練專用的Haar級聯分類器
實現基於手機攝影機的即時滑板車偵測
提供完整的展示視頻和技術文件
🚀 快速開始 Quick Start
1. 環境安裝
# 克隆專案
git clone https://github.com/your-username/campus-scooter-detection.git
cd campus-scooter-detection

# 安裝依賴
pip install -r requirements.txt

# 驗證安裝
python demo.py --check
2. 資料準備
# 查看當前資料統計
python data_collection.py --action stats

# 從攝影機收集資料
python data_collection.py --action camera --type positive --duration 120

# 從視頻收集資料
python data_collection.py --action video --source your_video.mp4 --type positive

# 匯入現有影像
python data_collection.py --action import --source /path/to/images --type positive
3. 資料標註
# 啟動LabelImg標註工具
python data_collection.py --action annotate

# 驗證標註品質
python data_collection.py --action validate
4. 訓練模型
# 驗證訓練資料
python train_cascade.py --validate_only

# 開始訓練（預計需要數小時）
python train_cascade.py --num_stages 20 --num_pos 1800 --num_neg 3600
5. 測試偵測
# 電腦攝影機偵測
python demo.py --webcam --cascade data/trained_model/cascade.xml

# 手機攝影機偵測
python demo.py --mobile --cascade data/trained_model/cascade.xml

# 影片檔案偵測
python demo.py --video test_video.mp4 --cascade data/trained_model/cascade.xml
📁 專案結構 Project Structure
campus-scooter-detection/
├── data/                           # 資料目錄
│   ├── positive/                   # 正樣本影像
│   ├── negative/                   # 負樣本影像
│   ├── annotations/                # 標註檔案
│   └── trained_model/              # 訓練後的模型
│       └── cascade.xml
├── src/                            # 核心程式碼
│   ├── scooter_detector.py         # 主偵測器
│   ├── mobile_camera.py            # 手機攝影機整合
│   ├── train_cascade.py            # 訓練腳本
│   └── data_collection.py          # 資料收集工具
├── utils/                          # 工具模組
│   ├── __init__.py
│   ├── annotation_helper.py        # 標註輔助工具
│   └── image_processing.py         # 影像處理工具
├── demo.py                         # 演示程式
├── requirements.txt                # 相依套件
├── setup.py                       # 安裝腳本
└── README.md                       # 說明文件
🔧 手機攝影機設定 Mobile Camera Setup
步驟 1: 安裝應用程式
在Android手機上安裝 IP Webcam 應用程式：

Google Play商店搜尋 "IP Webcam"
下載並安裝（由Pavel Khlebovich開發）
步驟 2: 網路連接
確保手機和電腦連接到相同的WiFi網路

步驟 3: 啟動服務
開啟IP Webcam應用程式
調整設定（解析度、品質等）
點擊 "Start Server"
記下顯示的IP位址
步驟 4: 連接測試
# 自動搜尋手機攝影機
python mobile_camera.py

# 或使用演示程式
python demo.py --mobile --cascade your_model.xml
📊 訓練參數說明 Training Parameters
資料需求
正樣本: 建議至少 200-500 張滑板車影像
負樣本: 建議至少 1000-2000 張背景影像
標註格式: YOLO格式 (class x_center y_center width height)
訓練參數
python train_cascade.py \
    --num_stages 20 \        # 訓練階段數（越多越精確但更慢）
    --num_pos 1800 \         # 正樣本數量
    --num_neg 3600 \         # 負樣本數量
    --width 24 \             # 偵測視窗寬度
    --height 24              # 偵測視窗高度
效能調優
scaleFactor: 1.05-1.3（較小值更精確但更慢）
minNeighbors: 3-8（較大值減少誤偵測）
minSize: 調整最小偵測尺寸
🔧 減少誤偵測的優化指南
訓練參數優化
高精確度訓練模式:

# 使用高精確度模式訓練
python train_cascade.py --high_precision --num_stages 15 --python_only

# 手動指定嚴格參數
python train_cascade.py \
    --num_stages 20 \
    --num_pos 15 \
    --num_neg 100 \
    --python_only
偵測參數調整
減少誤偵測的偵測參數:

# 在 demo.py 或其他偵測程式中使用這些參數
detector_params = {
    'scaleFactor': 1.15,        # 較大值，減少計算
    'minNeighbors': 6,          # 較大值，減少誤偵測
    'minSize': (40, 40),        # 過濾小物件
    'maxSize': (300, 300),      # 限制最大尺寸
    'flags': cv2.CASCADE_SCALE_IMAGE
}
資料改善策略
收集更好的負樣本:

# 收集包含常見誤偵測物件的負樣本
python data_collection.py --action camera --type negative --duration 300

# 從誤偵測的物件中提取負樣本
python data_collection.py --action import --source /path/to/false_positives --type negative
標註品質改善:

確保標註框緊貼滑板車邊界
避免包含過多背景
標註完整的滑板車（包含車輪、把手、踏板）
移除模糊或部分遮蔽的樣本
測試和驗證
階段性測試:

# 測試不同階段的模型
python demo.py --cascade data/trained_model/cascade.xml --test_mode

# 記錄誤偵測案例
python demo.py --cascade data/trained_model/cascade.xml --log_detections

# 批次測試多個影像
python demo.py --cascade data/trained_model/cascade.xml --batch_test /path/to/test/images
🎬 展示視頻錄製 Demo Video Recording
系統自動支援視頻錄製功能：

# 錄製展示視頻
python scooter_detector.py \
    --cascade data/trained_model/cascade.xml \
    --source 0 \
    --output demo_video.avi

# 手機攝影機錄製
python demo.py --mobile --cascade data/trained_model/cascade.xml
# 在執行過程中會自動錄製
📈 效能評估 Performance Evaluation
評估指標
精確度 (Precision): TP / (TP + FP)
召回率 (Recall): TP / (TP + FN)
F1分數: 2 × (Precision × Recall) / (Precision + Recall)
即時性: FPS（每秒處理幀數）
測試建議
準備多樣化的測試影像
記錄不同光照條件下的表現
測試不同角度和距離的偵測效果
統計誤偵測和漏偵測情況
🛠️ 故障排除 Troubleshooting
常見問題
Q: 找不到opencv_createsamples或opencv_traincascade

# 確認OpenCV安裝
python -c "import cv2; print(cv2.__version__)"

# 重新安裝OpenCV
pip uninstall opencv-python opencv-contrib-python
pip install opencv-contrib-python
Q: 手機攝影機連接失敗

檢查手機和電腦是否在同一網路
確認IP Webcam應用程式正在運行
嘗試在瀏覽器開啟 http://手機IP:8080
Q: 訓練過程中斷

檢查正負樣本數量是否足夠
確認標註檔案格式正確
降低訓練參數（num_stages, num_pos等）
Q: 偵測效果不佳

增加訓練資料多樣性
調整偵測參數（scaleFactor, minNeighbors）
重新檢查標註品質
📧 聯絡資訊 Contact
作者: [沈冠廷]
學號: [1120417]
Email: [s1120417@mail.yzu.edu.tw]
課程: EEB215A 電腦視覺與影像處理概論
GitHub: https://github.com/your-username/campus-scooter-detection
