% CVD2014
data_path = 'E:/VQADatabase/CVD2014/Realignment_MOS.csv'; % 修改為 CVD2014 的數據路徑
data = readtable(data_path);  % 加載 CVD2014 的數據

% 提取主觀評分
scores = arrayfun(@(i) str2double(data.RealignmentMOS{i}) / 100, 1:234)'; 

% 提取視頻名稱
% 動態生成視頻名稱
root_path = 'E:/VQADatabase/CVD2014';  % 定義根目錄
video_names = arrayfun(@(i) fullfile(root_path, ...
    ['Test' data.File_name{i}(6)], ... % Test 資料集目錄 (如 Test1、Test2)
    strrep(strrep(data.Content{i}, ':', '_'), ' ', '_'), ... % 替換 ':' 和空格
    [data.File_name{i} '.avi']), ... % 視頻名稱 (如 Test01_City_D01.avi)
    1:234, 'UniformOutput', false)';

% 初始化高度和寬度數組
heights = zeros(length(video_names), 1); 
widths = zeros(length(video_names), 1);

% 提取每個視頻的分辨率
for i = 1:length(video_names)
    video_path = video_names{i};  % 獲取視頻的完整路徑
    
    % 打印調試信息
    fprintf('正在處理第 %d 個視頻: %s\n', i, video_path);

    v = VideoReader(video_path);  % 使用 VideoReader 打開視頻
    heights(i) = v.Height;  % 獲取視頻的高度
    widths(i) = v.Width;    % 獲取視頻的寬度
end

% 定義其他參數
max_len = 830;  % 假設最大視頻幀數
video_format = 'RGB';  % 定義視頻格式
ref_ids = [1:length(scores)]';  % 每個視頻的參考 ID

% 隨機訓練、驗證、測試集的劃分，進行 1000 次隨機抽樣
index = cell2mat(arrayfun(@(i) randperm(length(scores)), ...
    1:1000, 'UniformOutput', false)');

% 保存結果到 .mat 文件，包括每個視頻的高、寬信息
save('CVD2014info', 'video_names', 'scores', 'heights', 'widths', 'max_len', 'video_format', 'ref_ids', 'index', '-v7.3');





