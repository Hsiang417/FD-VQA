% 假設你的 TXT 檔案路徑
txt_file_path = 'E:/VQADatabase/LSVQ/labels_1080p.txt'; % 修改為你的 TXT 檔案路徑
txt_data = readtable(txt_file_path, 'Delimiter', ',', 'ReadVariableNames', false);

% 提取影片名稱和主觀評分
video_names = txt_data.Var1; % 第一欄為影片名稱
scores = txt_data.Var4; % 第四欄為主觀評分

% 初始化高度和寬度
heights = zeros(length(video_names), 1);
widths = zeros(length(video_names), 1);

% 提取每部影片的分辨率
for i = 1:length(video_names)
    video_path = fullfile('E:/VQADatabase/LSVQ/', video_names{i}); % 修改影片根目錄
    if isfile(video_path)
        v = VideoReader(video_path); % 使用 VideoReader 打開影片
        heights(i) = v.Height;
        widths(i) = v.Width;
    else
        warning('影片未找到: %s', video_names{i});
        heights(i) = -1; % 若無法找到，給予 -1 表示
        widths(i) = -1;
    end
end

% 假設其他參數
max_len = 900; % 假設最大幀數
video_format = 'RGB'; % 定義影片格式
ref_ids = (1:length(scores))'; % 每部影片的參考 ID

% 隨機訓練、驗證、測試集的劃分，進行1000次隨機抽樣
index = cell2mat(arrayfun(@(i) randperm(length(scores)), 1:1000, 'UniformOutput', false)');

% 保存結果為 .mat 文件
save('LSVQ1080p_test_info.mat', 'video_names', 'scores', 'heights', 'widths', 'max_len', 'video_format', 'ref_ids', 'index', '-v7.3');
