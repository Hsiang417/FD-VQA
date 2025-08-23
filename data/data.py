import numpy as np
import pandas as pd
import cv2
import os
from scipy.io import savemat

# 設定 MOS Excel 檔案和 CSV 特徵檔案的路徑
data_path = 'E:/VQADatabase/YoutubeUGC/original_videos/original_videos/MOS_for_YouTube_UGC_dataset.xlsx'
feature_path = 'E:/VQADatabase/YoutubeUGC/original_videos/original_videos/dataset_selection_features.csv'
video_root_folder = 'E:/VQADatabase/YoutubeUGC/original_videos/original_videos/'  # 設定影片的根目錄
mos_sheet = 'MOS'  # Excel 中的工作表名稱

# 讀取 MOS 資料
mos_data = pd.read_excel(data_path, sheet_name=mos_sheet)
video_names = mos_data['vid'].values  # 提取影片名稱（請確認欄位名稱）
scores = mos_data['MOSfull'].values  # 提取 MOS 分數（請確認欄位名稱）

# 讀取特徵資料，包含寬度和高度資訊
feature_data = pd.read_csv(feature_path)

# 將特徵資料和 MOS 資料合併，使用 `vid` 與 `FILENAME` 作為對應關鍵
merged_data = pd.merge(mos_data, feature_data, left_on='vid', right_on='FILENAME', how='inner')

# 過濾掉找不到高度、寬度或影片檔案的影片，並提取高度、寬度和幀數資訊
valid_video_names = []
valid_scores = []
heights = []
widths = []
frame_counts = []


# 定義一個函數來遞迴搜尋影片文件
def find_video_path(root_folder, video_name):
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.startswith(video_name) and file.endswith('.mkv'):
                return os.path.join(root, file)
    return None


for idx, row in merged_data.iterrows():
    if not pd.isna(row['HEIGHT']) and not pd.isna(row['WIDTH']):
        # 嘗試找到影片的完整路徑
        video_name = row['vid']
        video_path = find_video_path(video_root_folder, video_name)

        if video_path:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_counts.append(frame_count)
                cap.release()
            else:
                print(f"無法讀取影片: {video_path}")
                continue  # 如果無法讀取影片，跳過該影片
        else:
            print(f"找不到影片: {video_name}")
            continue

        # 如果成功找到並讀取影片，則將相關資訊加入有效列表
        valid_video_names.append(video_name)
        valid_scores.append(row['MOSfull'])
        heights.append(row['HEIGHT'])
        widths.append(row['WIDTH'])

# 計算 max_len 作為找到的影片的最大幀數
max_len = max(frame_counts) if frame_counts else 0

# 轉換為 numpy 陣列，並調整形狀為 1024x1
heights = np.array(heights).reshape(-1, 1)
widths = np.array(widths).reshape(-1, 1)
scores = np.array(valid_scores).reshape(-1, 1)
ref_ids = np.arange(1, len(valid_scores) + 1).reshape(-1, 1)
video_names = np.array(valid_video_names, dtype=object).reshape(-1, 1)  # 使用 dtype=object 來轉換為 Cell 格式

# 隨機生成訓練、驗證、測試集的分割索引，執行1000次
index = np.array([np.random.permutation(len(valid_scores)) for _ in range(1000)])

# 準備數據結構以保存到 .mat 檔案，按照指定的順序排列
output_data = {
    'height': heights,
    'index': index,
    'max_len': max_len,
    'ref_ids': ref_ids,
    'scores': scores,
    'video_format': 'RGB',
    'video_names': video_names,  # video_names 已轉為 1024x1 cell 格式
    'width': widths
}

# 儲存為 .mat 檔案
output_path = 'YouTube_UGC_info.mat'
savemat(output_path, output_data)
print(f'資料已儲存到 {output_path}')
