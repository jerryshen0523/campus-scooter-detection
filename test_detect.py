import cv2
import glob
import os

# 設定模型路徑與正樣本資料夾
cascade_path = r'data\trained_model\cascade.xml'  # 你的模型檔案路徑
positive_dir = r'data\positive'                  # 正樣本資料夾

# 載入模型
detector = cv2.CascadeClassifier(cascade_path)

# 取得所有正樣本影像
image_paths = glob.glob(os.path.join(positive_dir, '*.*'))

for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 偵測物件
    results = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    # 畫出偵測結果
    for (x, y, w, h) in results:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 顯示影像
    cv2.imshow('Detection', img)
    print(f"{os.path.basename(img_path)} 偵測到 {len(results)} 個物件")
    cv2.waitKey(0)  # 按任意鍵看下一張

cv2.destroyAllWindows()