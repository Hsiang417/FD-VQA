% === 設定路徑與檔案參數 ===
data_path = 'E:/VQADatabase/KoNViD_1k/KoNViD_1k_attributes.csv';  % CSV 檔案路徑
root_path = 'E:/VQADatabase/KoNViD_1k/';                         % 存放 KoNViD_1k 影片的資料夾

% === 讀取 CSV ===
T = readtable(data_path);         % 讀取屬性表
% 假設表格 T 內有欄位：'file_name' 與 'MOS'
video_names_raw = T.file_name;    % 提取影片名稱 (不含路徑)
scores          = T.MOS;          % 提取 MOS 主觀評分

% === 動態生成完整影片路徑 ===
num_videos  = length(video_names_raw);
video_names = cell(num_videos, 1); % 預留空間
for i = 1:num_videos
    % 將「檔名 + .mp4 或 .avi」與 root_path 組合成完整路徑
    % (實際副檔名請依你檔案狀況修正)
    video_names{i} = fullfile(root_path, [video_names_raw{i}]);
end

% === 初始化高度和寬度 ===
heights = zeros(num_videos, 1);
widths  = zeros(num_videos, 1);

% === 逐一讀取影片分辨率 ===
for i = 1:num_videos
    video_path = video_names{i};
    if ~isfile(video_path)
        fprintf('警告：找不到檔案 %s\n', video_path);
        continue;  % 若找不到檔案，可選擇跳過或中斷
    end

    % 透過 VideoReader 讀取影片
    v = VideoReader(video_path);
    heights(i) = v.Height;
    widths(i)  = v.Width;

    fprintf('處理第 %d/%d 個影片: %s (W=%d, H=%d)\n', ...
        i, num_videos, video_path, widths(i), heights(i));
end

% === 定義其他參數 ===
max_len       = 830;       % 假設最大幀數 (可依 KoNViD_1k 的數據做統計)
video_format  = 'RGB';     % 影片的像素格式
ref_ids       = (1:num_videos)';  % 參考 ID，簡單設定為 1~N

% === 隨機劃分資料集 (範例: 1000 次隨機抽樣) ===
num_splits = 1000;
index = zeros(num_videos, num_splits);
for s = 1:num_splits
    index(:, s) = randperm(num_videos);
end

% === 儲存到 KoNViD_1k_info.mat ===
save('KoNViD_1k_info.mat', ...
    'video_names', 'scores', 'heights', 'widths', ...
    'max_len', 'video_format', 'ref_ids', 'index', ...
    '-v7.3');

disp('已成功產生 KoNViD_1k_info.mat');
