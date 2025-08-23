% LIVE-VQC
data_path = 'D:/VQAdatabase/LIVE Video Quality Challenge (VQC) Database/data.mat'; % 修改為你的 LIVE-VQC 資料集路徑
data = load(data_path);  % 加載 LIVE-VQC 的數據
scores = data.mos;  % 提取主觀評分
video_names = data.video_list;  % 提取視頻名稱

% 初始化高度和寬度數組
heights = zeros(length(video_names), 1); 
widths = zeros(length(video_names), 1);

% 提取每個視頻的分辨率
for i = 1:length(video_names)
    video_path = fullfile('D:/VQAdatabase/LIVE Video Quality Challenge (VQC) Database/Video', video_names{i}); % 拼接視頻路徑
    v = VideoReader(video_path);  % 使用 VideoReader 打開視頻
    heights(i) = v.Height;  % 獲取視頻的高度
    widths(i) = v.Width;    % 獲取視頻的寬度
end

max_len = 900; % 假設最大視頻幀數
video_format = 'RGB';  % 定義視頻格式
ref_ids = [1:length(scores)]';  % 每個視頻的參考 ID

% 隨機訓練、驗證、測試集的劃分，進行1000次隨機抽樣
index = cell2mat(arrayfun(@(i)randperm(length(scores)), ...
    1:1000,'UniformOutput', false)'); 

% 保存結果到 .mat 文件，包括每個視頻的高、寬信息
save('LIVE-VQCinfo', 'video_names', 'scores', 'heights', 'widths', 'max_len', 'video_format', 'ref_ids', 'index', '-v7.3');