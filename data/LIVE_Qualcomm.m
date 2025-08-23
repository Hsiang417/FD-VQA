% LIVE-Qualcomm
data_path = 'E:/VQADatabase/LIVE-QualcommDatabase/qualcommSubjectiveData.mat';
data = load(data_path);

% 提取主觀評分
scores = data.qualcommSubjectiveData.unBiasedMOS; 

% 提取視頻名稱並組合資料夾
distortionNames = data.qualcommVideoData.distortionNames; % 6 個資料夾名稱
distortionType = data.qualcommVideoData.distortionType;   % 每個視頻的資料夾索引
vidNames = data.qualcommVideoData.vidNames;              % 每個視頻的名稱

% 動態生成視頻路徑
video_names = arrayfun(@(i) fullfile(distortionNames{distortionType(i)}, vidNames{i}), ...
    1:length(vidNames), 'UniformOutput', false)';

% 初始化高度和寬度數組
heights = zeros(length(video_names), 1); 
widths = zeros(length(video_names), 1);

% 自定義讀取 YUV 文件的函數
function frames = readYUV(video_path, width, height)
    fid = fopen(video_path, 'r');
    if fid == -1
        error('無法打開文件: %s', video_path);
    end

    frame_size = width * height * 1.5; % YUV420 格式
    fseek(fid, 0, 'eof');
    file_size = ftell(fid);
    num_frames = floor(file_size / frame_size);
    fseek(fid, 0, 'bof');

    frames = cell(1, num_frames);

    for i = 1:num_frames
        Y = fread(fid, [width, height], 'uint8')';
        U = fread(fid, [width / 2, height / 2], 'uint8')';
        V = fread(fid, [width / 2, height / 2], 'uint8')';

        frames{i} = yuv2rgb(Y, U, V);
    end

    fclose(fid);
end

function rgb = yuv2rgb(Y, U, V)
    Y = double(Y);
    U = imresize(double(U) - 128, 2);
    V = imresize(double(V) - 128, 2);

    R = Y + 1.402 * V;
    G = Y - 0.344136 * U - 0.714136 * V;
    B = Y + 1.772 * U;

    rgb = cat(3, uint8(R), uint8(G), uint8(B));
end

% 處理所有視頻
for i = 1:length(video_names)
    video_path = fullfile('E:/VQADatabase/LIVE-QualcommDatabase/Videos', video_names{i});
    [~, ~, ext] = fileparts(video_path); % 獲取文件擴展名

    fprintf('Processing video %d/%d: %s\n', i, length(video_names), video_path);

    if strcmp(ext, '.yuv') % 處理 YUV 文件
        try
            frames = readYUV(video_path, 1920, 1080); % 替換為實際寬高
            heights(i) = size(frames{1}, 1);
            widths(i) = size(frames{1}, 2);
        catch ME
            warning('Error reading YUV file: %s\nMessage: %s\n', video_path, ME.message);
            continue;
        end
    else % 處理其他支持的格式
        try
            v = VideoReader(video_path);
            heights(i) = v.Height;
            widths(i) = v.Width;
        catch ME
            warning('Error reading video: %s\nMessage: %s\n', video_path, ME.message);
            continue;
        end
    end
end

% 定義其他參數
max_len = 526;  % 假設最大視頻幀數
video_format = 'RGB';  % 定義視頻格式
ref_ids = [1:length(scores)]';  % 每個視頻的參考 ID

% 隨機訓練、驗證、測試集的劃分，進行 1000 次隨機抽樣
index = cell2mat(arrayfun(@(i) randperm(length(scores)), ...
    1:1000, 'UniformOutput', false)');

% 保存結果到 .mat 文件
save('LIVE-Qualcomm_info', 'video_names', 'scores', 'heights', 'widths', 'max_len', 'video_format', 'ref_ids', 'index', '-v7.3');

