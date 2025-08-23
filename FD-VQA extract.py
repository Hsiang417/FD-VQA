import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import convnext_base
from torch.utils.data import Dataset
from PIL import Image
import os
import h5py
import numpy as np
import random
from argparse import ArgumentParser
import skvideo.io
import pandas as pd
import torch.nn.functional as F
import glob
import shutil
import subprocess
import torchvision.models as models
import cv2
import time
from scipy.io import loadmat  # 加在檔案開頭

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



class VideoDataset(Dataset):
    def __init__(self, videos_dir, video_names, scores, video_format='RGB', widths=None, heights=None):
        super(VideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.video_names = video_names
        self.scores = scores
        self.format = video_format

        # 確保 widths 和 heights 是列表，且長度與 video_names 一致
        if widths is None or heights is None:
            raise ValueError("Widths and heights must be provided as lists.")
        if len(widths) != len(video_names) or len(heights) != len(video_names):
            raise ValueError("Lengths of widths, heights, and video_names must match.")

        self.widths = widths
        self.heights = heights

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        if self.videos_dir is None:
            video_path = video_name  # 已是完整路徑
        else:
            video_path = os.path.join(self.videos_dir, video_name)

        video_score = self.scores[idx]
        width = self.widths[idx]
        height = self.heights[idx]

        try:
            if video_path.endswith('.yuv'):
                if not isinstance(width, int) or not isinstance(height, int):
                    raise ValueError(f"Width and height must be integers for video at index {idx}")
                video_data = skvideo.io.vread(
                    video_path,
                    inputdict={
                        '-pix_fmt': 'yuv420p',
                        '-s': f'{width}x{height}'
                    }
                )
            else:
                video_data = skvideo.io.vread(video_path, outputdict={'-pix_fmt': 'rgb24'})

                # --- 防記憶體爆炸：隨機抽幀並縮圖 ---
                max_frames = 800
                resize_size = (1080, 1920)
                if video_data.shape[0] > max_frames:
                    video_data = skvideo.io.vread(video_path, outputdict={'-pix_fmt': 'rgb24'})

                    # 每 2 幀取一幀
                    video_data = video_data[::2]

                    # 不壓縮解析度
            video_tensor = torch.stack([
                        self.transform(Image.fromarray(frame)) for frame in video_data
                    ])


        except Exception as e:
            raise ValueError(f"❌ 讀取或轉換影片失敗: {video_path}，錯誤訊息：{e}")

        return {'video': video_tensor, 'score': video_score}




def global_std_pool2d(x):
    """2D 全域標準偏差池化"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1), dim=2, keepdim=True)

class ConvNeXtFeatureExtractor(nn.Module):
    """修改過的 ConvNeXt 用於特徵提取"""
    def __init__(self, swin_window_size=7):
        super(ConvNeXtFeatureExtractor, self).__init__()
        self.model = convnext_base(weights="DEFAULT")
        self.features = nn.Sequential(*list(self.model.children())[:-1])
        for p in self.features.parameters():
            p.requires_grad = False


    def forward(self, x):
        feature_maps = []
        feature_maps_att = []
        for ii, model in enumerate(self.features[0]):
            x = model(x)
            if ii in [3,5,6]:
                feature_maps.append(x)




        return feature_maps

class preConvNeXtFeatureExtractor(nn.Module):
    def __init__(self):
        super(preConvNeXtFeatureExtractor, self).__init__()
        self.model = convnext_base(weights="DEFAULT")
        self.features = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        for ii, layer in enumerate(self.features[0]):
            # print("layer:", ii, ":", x.shape)
            x = layer(x)
            if ii == 2:
                features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
                break
        return features_mean




def compute_weights(diff, alpha=1.0):
    return torch.exp(alpha * diff * 1)

def get_weights(diffs, alpha=1.0):
    weights = compute_weights(diffs, alpha)
    weights = torch.cat([torch.ones(1, device=weights.device), weights])
    print("weights:", weights.shape)
    return weights

def apply_weights_to_features(frame_features, weights):
    print(f"frame_features shape: {frame_features.shape}")
    weights = weights.unsqueeze(1)
    print(f"weights shape: {weights.shape}")
    if weights.shape[0] != frame_features.shape[0]:
        raise ValueError(f"The size of weights ({weights.shape[0]}) must match the size of frame_features ({frame_features.shape[0]}) at non-singleton dimension 0")
    # weighted_frame_features = frame_features * weights
    weighted_frame_features = frame_features
    return weighted_frame_features

def get_features_and_diffs(video_data, device):

    extractor1 = preConvNeXtFeatureExtractor().to(device)
    video_length = video_data.shape[0]
    frame_features = []
    diffs = []

    extractor1.eval()
    with torch.no_grad():
        for frame_idx in range(video_length):
            frame = video_data[frame_idx].unsqueeze(0).to(device)
            current_features_mean = extractor1(frame)
            # frame_features.append(current_features_mean.flatten(1))

            if frame_idx > 0:
                diff = torch.norm(current_features_mean - prev_features_mean, p=2)

                diffs.append(diff)


            prev_features_mean = current_features_mean
    if not diffs:
        diffs.append(torch.tensor(0.0, device=device))
    # frame_features = torch.cat(frame_features, dim=0)
    diffs = torch.tensor(diffs, device=device)

    return  diffs



def canny(frame, low_threshold=100, high_threshold=200):
    # frame 的形狀應該是 [H, W, 3]，即 [Height, Width, RGB channels]
    # if isinstance(frame, torch.Tensor):
    #     frame = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, C]

    # 分別提取 R, G, B 通道
    r_channel = frame[:, :, 0]
    g_channel = frame[:, :, 1]
    b_channel = frame[:, :, 2]

    # 分別將 R, G, B 通道轉換為灰階並應用 Canny 邊緣檢測
    r_edges = cv2.Canny(r_channel.astype(np.uint8), low_threshold, high_threshold)
    g_edges = cv2.Canny(g_channel.astype(np.uint8), low_threshold, high_threshold)
    b_edges = cv2.Canny(b_channel.astype(np.uint8), low_threshold, high_threshold)

    # 堆疊三個邊緣檢測結果，形成 [3, H, W] 的張量
    edges_stacked = np.stack([r_edges, g_edges, b_edges], axis=0)  # [3, H, W]

    # 將結果轉換為 PyTorch 張量並添加 batch 維度，形狀為 [1, 3, H, W]
    edges_tensor = torch.tensor(edges_stacked, dtype=torch.float32).unsqueeze(0)  # [1, 3, H, W]

    return edges_tensor
def compute_motion_region(frame1_gray, frame2_gray, frame_np, threshold_factor=1.5):
    # 計算光流
    flow = cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, None,
                                        pyr_scale=0.5, levels=5, winsize=25,
                                        iterations=5, poly_n=7, poly_sigma=1.5, flags=0)
    # 計算光流的幅度和方向
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 設定門檻來檢測運動區域
    threshold = np.mean(mag) + threshold_factor * np.std(mag)
    motion_mask = mag > threshold

    # 找出運動區域的邊界
    ys, xs = np.where(motion_mask)
    if ys.size == 0 or xs.size == 0:
        print("未檢測到運動區域")
        return None  # 或者處理沒有運動區域的情況

    xmin = xs.min()
    xmax = xs.max()
    ymin = ys.min()
    ymax = ys.max()

    # 裁剪原始圖像中對應的部分
    cropped_region = frame_np[ymin:ymax+1, xmin:xmax+1, :]

    # 將裁剪得到的圖像區域放大到原始圖像的尺寸
    resized_region = cv2.resize(cropped_region,
                                (frame_np.shape[1], frame_np.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
    return resized_region


def get_additional_features(current_video, device):
    diffs = get_features_and_diffs(current_video, device)
    print(diffs.shape)
    if torch.isnan(diffs).any() or torch.isinf(diffs).any():
        print("NaN or Inf detected in diffs.")
    diffs_mean = torch.mean(diffs).item()
    print(f'Average diff for video {i}: {diffs_mean}')

    # 將 diffs 轉為 numpy 格式
    diffs_np = diffs.cpu().numpy()

    # 檢查是否有足夠的影格數來計算 80% 門檻
    if len(diffs_np) < 6:  # 假設至少需要 5 個影格來計算百分比，這個值可以調整
        print(f'Not enough frames to select top 80% diffs for video {i}. Selecting all frames.')
        selected_indices = np.arange(len(diffs_np))  # 選取所有影格
        diffs = torch.ones_like(diffs)  # 將所有差異值設為 1，但保持張量格式一致

    else:
        threshold = np.percentile(diffs_np, 80)
        print(f'Diff threshold for top 20% for video {i}: {threshold}')
        selected_indices = np.where(diffs_np > threshold)[0]  # 選取大於門檻的影格


    print(f'Selected indices for video {i}: {selected_indices}')


    # if selected_indices.size == 0:
    #     print("沒有選取到 diff > 0.80 的影格")
    #     return None, None, None

    extractor2 = ConvNeXtFeatureExtractor().to(device)  # 特徵提取器2

    all_features = []  # 保存所有特徵
    all_indices = []  # 保存所有索引
    unified_feature_size = (16, 30)  # 統一的特徵尺寸

    with torch.no_grad():

        for idx in selected_indices:
            if idx == 0:

                if current_video.shape[0] == 1:  # 處理只有一幀的情況
                    print("影片只有一幀，無法選取相鄰幀。")
                    valid_indices = [0,0]  # 使用單一幀重複
                    print(f"處理單幀，使用有效索引: {valid_indices}")
                else:
                    valid_indices = [0, 0, 1]

            elif idx == current_video.shape[0] - 1:
                valid_indices = [current_video.shape[0] - 2, current_video.shape[0] - 1, current_video.shape[0] - 1]
            else:
                valid_indices = [idx + offset for offset in [-1, 0, 1] if 0 <= idx + offset < current_video.shape[0]]

            print(f"處理 idx {idx}, 有效索引: {valid_indices}")

            prev_frame_gray = None  # 初始化前一幀為 None

            for new_idx in valid_indices:


                # 使用原始圖像數據
                frame = current_video[new_idx].unsqueeze(0).to(device)
                if torch.isnan(frame).any() or torch.isinf(frame).any():
                    print(f"NaN or Inf detected in frame at index {new_idx}. Replacing values.")
                # print(f"frame shape: {frame.shape}")

                # 將 frame 轉換為可以被 OpenCV 處理的格式
                frame_np = frame.squeeze(0).cpu().numpy().transpose(1, 2, 0)

                frame_gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)

                # 如果 prev_frame_gray 為 None，則初始化
                if prev_frame_gray is None:
                    prev_frame_gray = frame_gray
                    continue



                # 調用函數計算運動區域
                resized_region = compute_motion_region(prev_frame_gray, frame_gray, frame_np, threshold_factor=1.5)

                # 更新 prev_frame_gray
                prev_frame_gray = frame_gray


                    # continue

                # # 顯示結果
                # # 計算光流（Farneback 方法）
                # flow = cv2.calcOpticalFlowFarneback(
                #     prev_frame_gray, frame_gray, None,
                #     pyr_scale=0.5, levels=5, winsize=25,
                #     iterations=5, poly_n=7, poly_sigma=1.5, flags=0
                # )
                #
                # # 計算光流方向和幅度
                # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                #
                # # 生成 HSV 光流圖
                # hsv = np.zeros((frame_gray.shape[0], frame_gray.shape[1], 3), dtype=np.uint8)  # 確保 HSV 大小與灰度圖一致
                # hsv[..., 1] = 255  # 飽和度設為最大值
                # hsv[..., 0] = ang * 180 / np.pi / 2  # 方向對應 Hue
                # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # 幅度對應亮度

                # 將 HSV 轉換為 RGB 格式
                # flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

                # # 顯示結果
                # plt.figure(figsize=(15, 5))
                #
                # # 顯示原始幀
                # plt.subplot(1, 3, 1)
                # plt.imshow(np.clip(frame_np, 0, 1))
                # plt.title(f"Original frame {new_idx}")
                #
                # # 顯示放大後的裁剪區域
                # if resized_region is not None:
                #     plt.subplot(1, 3, 2)
                #     plt.imshow(np.clip(resized_region, 0, 1))
                #     plt.title(f"Motion region replaced frame {new_idx}")
                #
                # # 顯示光流圖
                # plt.subplot(1, 3, 3)
                # plt.imshow(flow_rgb)
                # plt.title(f"Optical Flow visualization {new_idx}")
                #
                # plt.show()



                if resized_region is None:
                    print(f"在幀 {new_idx} 中未檢測到運動區域")
                    resized_region=frame
                    extracted_replaced_frame_features = extractor2(resized_region)

                else:
                    resized_region_tensor = torch.from_numpy(resized_region.transpose(2, 0, 1)).unsqueeze(0).float().to(
                        device)
                    extracted_replaced_frame_features = extractor2(resized_region_tensor)

                # vit_feature_frame = transformer_extractor(frame)
                # vit_feature_resized = transformer_extractor(resized_region_tensor, vit_extractor.model, device)

                processed_features_replaced_frame = []
                upsampled_maps = []
                for fmap in extracted_replaced_frame_features:
                    # 1. 先把空間尺寸對齊
                    fmap_upsampled = F.interpolate(
                        fmap,
                        size=unified_feature_size,
                        mode='bilinear',
                        align_corners=False
                    )

                    # 2. 基本數值安全檢查
                    if torch.isnan(fmap_upsampled).any() or torch.isinf(fmap_upsampled).any():
                        print("NaN or Inf detected in fmap_upsampled.")

                    # 3. 收集起來，稍後一次做統計
                    upsampled_maps.append(fmap_upsampled)

                # 4. 在 channel 維度 (dim=1) 串接成一張「大特徵圖」
                #    如果一共有 N 個 stage，而每張 fmap channel 為 C，
                #    這邊 concat 後 channel 就是 C_total = N * C
                concat_fmaps = torch.cat(upsampled_maps, dim=1)  # [B, C_total, H_u, W_u]

                # 5. 一次性計算 mean 與 std
                features_canny_mean = nn.functional.adaptive_avg_pool2d(concat_fmaps, 1)  # [B, C_total, 1, 1]
                features_canny_std = global_std_pool2d(concat_fmaps)  # [B, C_total, 1, 1] (或 [B, C_total])

                # 6. 攤平成向量並串在一起（mean | std）
                processed_features_replaced_frame = torch.cat(
                    [
                        features_canny_mean.view(features_canny_mean.size(0), -1),  # [B, C_total]
                        features_canny_std.view(features_canny_std.size(0), -1)  # [B, C_total]
                    ],
                    dim=1
                )  # ➜ [B, 2 * C_total]







                # 提取 Canny 邊緣特徵
                frame_canny = frame.squeeze(0).permute(1, 2, 0)

                canny_edge = torch.tensor(canny(frame_canny.cpu().numpy()), dtype=torch.float32).to(device)

                # 使用 ConvNeXt 提取 Canny 特徵
                extracted_canny_features = extractor2(canny_edge)  # 增加 batch 維度

                # 處理 Canny 特徵
                processed_features_canny = []
                upsampled_canny_maps = []

                for fmap in extracted_canny_features:
                    # 1. 對齊空間尺寸
                    fmap_upsampled = F.interpolate(
                        fmap,
                        size=unified_feature_size,
                        mode='bilinear',
                        align_corners=False
                    )

                    # 2. 數值檢查
                    if torch.isnan(fmap_upsampled).any() or torch.isinf(fmap_upsampled).any():
                        print("NaN or Inf detected in features_canny fmap_upsampled.")

                    # 3. 收集
                    upsampled_canny_maps.append(fmap_upsampled)

                # 4. channel 維度 concat → [B, C_total, H_u, W_u]
                concat_canny = torch.cat(upsampled_canny_maps, dim=1)

                # 5. 一次算 mean / std
                canny_mean = nn.functional.adaptive_avg_pool2d(concat_canny, 1)  # [B, C_total, 1, 1]
                canny_std = global_std_pool2d(concat_canny)  # [B, C_total, 1, 1]

                # 6. 攤平 + 拼接 → [B, 2 * C_total]
                processed_features_canny = torch.cat(
                    [
                        canny_mean.view(canny_mean.size(0), -1),
                        canny_std.view(canny_std.size(0), -1)
                    ],
                    dim=1
                )

                # 處理原始特徵
                feature_maps = extractor2(frame)
                processed_features = []
                upsampled_maps = []

                for fmap in feature_maps:
                    # 1. 先把空間尺寸統一
                    fmap_upsampled = F.interpolate(
                        fmap,
                        size=unified_feature_size,
                        mode='bilinear',
                        align_corners=False
                    )

                    # 2. 數值檢查
                    if torch.isnan(fmap_upsampled).any() or torch.isinf(fmap_upsampled).any():
                        print("NaN or Inf detected in processed_features fmap_upsampled.")

                    # 3. 收集
                    upsampled_maps.append(fmap_upsampled)

                # 4. channel 維度 concat（假設總 channel 為 C_total）
                concat_maps = torch.cat(upsampled_maps, dim=1)  # [B, C_total, H_u, W_u]

                # 5. 一次性計算 mean / std
                features_mean = nn.functional.adaptive_avg_pool2d(concat_maps, 1)  # [B, C_total, 1, 1]
                features_std = global_std_pool2d(concat_maps)  # [B, C_total, 1, 1]

                # 6. 攤平 + 拼接 → [B, 2 * C_total]
                processed_features = torch.cat(
                    [
                        features_mean.view(features_mean.size(0), -1),
                        features_std.view(features_std.size(0), -1)
                    ],
                    dim=1
                )

                # 7. 與其他兩組特徵合併
                combined_features = torch.cat(
                    (processed_features, processed_features_canny, processed_features_replaced_frame),
                    dim=1
                )





                all_features.append(combined_features)
                all_indices.append(new_idx)

        all_features_tensor = torch.cat(all_features, dim=0)

        weights = get_weights(diffs)

        all_weights = weights[all_indices]

        weighted_frame_features = apply_weights_to_features(all_features_tensor, all_weights)


        print(f'Weighted frame features shape: {weighted_frame_features.shape}')
    return weighted_frame_features, torch.tensor(all_indices, device=device),diffs


def global_std_pool3d(x):
    """3D 全局标准差池化"""
    return torch.std(x.view(x.size(0), x.size(1), -1), dim=2, keepdim=True)

def exract3D(current_video, device):
    model = models.video.r3d_18(pretrained=True).to(device)
    T = 16
    batches = torch.split(current_video, T, dim=0)  # Split along the time dimension

    all3D_features = []

    for j, batch in enumerate(batches):
        print(f'Processing batch {j + 1}/{len(batches)}: shape {batch.shape}')

        # Batch shape should be [T, C, H, W], so add a batch dimension
        current_video3D = batch.unsqueeze(0)  # [1, T, C, H, W]
        current_video3D = current_video3D.permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]

        # Ensure data is on the correct device
        current_video3D = current_video3D.to(device)


        # Extract 3D convolutional features without final pooling
        with torch.no_grad():  # Avoid gradient computation to save memory
            feature3D = model(current_video3D)
        if torch.isnan(feature3D).any() or torch.isinf(feature3D).any():
            print("NaN or Inf detected in feature3D.")
        print(f'Feature map shape: {feature3D.shape}')
        all3D_features.append(feature3D.cpu())



    final3D_features = torch.cat(all3D_features, dim=0)

    print(f'3D Final feature shape: {final3D_features.shape}')
    final3Dmean_features = torch.mean(final3D_features, dim=0)
    final3Dmax_features = torch.max(final3D_features, dim=0).values
    if final3D_features.shape[0] == 1:
        print("Only one data point available, setting std to 0.")
        final3Dstd_features = torch.zeros_like(final3Dmean_features)
    else:
        final3Dstd_features = torch.std(final3D_features, dim=0)
    #
    #
    final3D_features = torch.cat((final3Dmean_features, final3Dmax_features,final3Dstd_features), dim=0)


    final3D_features = final3D_features.unsqueeze(0)

    print(f'3D Final feature shape: {final3D_features.shape}')
    return final3D_features

if __name__ == "__main__":
    parser = ArgumentParser(description='Use pretrained ConvNeXt to detect scene changes')
    parser.add_argument("--seed", type=int, default=19990417)
    parser.add_argument('--database', default='KoNViD-1k', type=str, help='Database name (default: KoNViD-1k)')
    parser.add_argument('--frame_batch_size', type=int, default=128, help='Frame batch size for feature extraction (default: 64)')
    parser.add_argument('--disable_gpu', action='store_true', help='Flag to disable GPU')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.database == 'KoNViD-1k':
        videos_dir = 'D:/KoNViD_1k/'
        features_dir = 'CNN_features_KoNViD-1k/'
        datainfo = 'data/KoNViD_1k_info.mat'
    elif args.database == 'CVD2014':
        videos_dir = 'E:/VQADatabase/CVD2014/'
        features_dir = 'CNN_features_CVD2014/'
        datainfo = 'data/CVD2014info.mat'
    elif args.database == 'LIVE-Qualcomm':
        videos_dir = 'E:\VQADatabase\LIVE-QualcommDatabase/videos/'
        features_dir = 'CNN_features_LIVE-Qualcomm/'
        datainfo = 'data/LIVE-Qualcomm_info.mat'
    elif args.database == 'LIVE-VQC':
        videos_dir = 'D:/VQAdatabase/LIVE Video Quality Challenge (VQC) Database/Video/'
        features_dir = 'CNN_features_LIVE-VQC/'
        datainfo = 'data/LIVE-VQCinfo.mat'
    elif args.database == 'Youtube_UGC':
        videos_dir = 'E:/VQADatabase/YoutubeUGC/original_videos/original_videos/'
        features_dir = 'CNN_features_Youtube_UGC/'
        datainfo = 'data/YouTubeUGC_info_valid_only.mat'
    elif args.database == 'LSVQ':
        videos_dir = 'E:/VQADatabase/LSVQ/'
        features_dir = 'CNN_features_LSVQ/'
        datainfo = 'data/LSVQ_info.mat'
    elif args.database == 'LSVQ_test':
        videos_dir = 'E:/VQADatabase/LSVQ/'
        features_dir = 'CNN_features_LSVQ_test/'
        datainfo = 'data/LSVQ_test_info.mat'
    elif args.database == 'LSVQ1080p_test':
        videos_dir = 'E:/VQADatabase/LSVQ/'
        features_dir = 'CNN_features_LSVQ1080p_test/'
        datainfo = 'data/LSVQ1080p_info.mat'



    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    if args.database == 'Youtube_UGC':
        videos_dir = 'E:/VQADatabase/YoutubeUGC/original_videos/original_videos/'
        features_dir = 'CNN_features_Youtube_UGC/'
        datainfo = 'data/YouTubeUGC_info_valid_only.mat'

        mat_data = loadmat(datainfo)

        # 原始影片名稱（例如 'Animation_1080P-05f8'）
        raw_video_names = [str(v[0]) for v in mat_data['video_names'].squeeze()]
        scores = mat_data['scores'].squeeze()
        widths = mat_data['widths'].squeeze().tolist()
        heights = mat_data['heights'].squeeze().tolist()
        video_format = mat_data['video_format'][0]  # 例如 'RGB'

        # 產生完整路徑
        video_names = []
        for name in raw_video_names:
            parts = name.split('_')
            if len(parts) >= 2:
                category = parts[0]
                resolution = parts[1].split('-')[0]
                full_path = os.path.join(videos_dir, category, resolution, name + '.mkv')
                video_names.append(full_path)
            else:
                raise ValueError(f"Invalid video name format: {name}")

        # 建立 Dataset，直接傳入完整路徑
        dataset = VideoDataset(
            videos_dir=None,  # 不需要用了
            video_names=video_names,  # 已是完整路徑
            scores=scores,
            widths=widths,
            heights=heights
        )

    else:
        with h5py.File(datainfo, 'r') as Info:
            video_names = [
                Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode('utf-8')
                for i in range(len(Info['video_names'][0, :]))
            ]
            scores = Info['scores'][0, :]
            video_format = Info['video_format'][()].tobytes()[::2].decode('utf-8')
            widths = Info['widths'][0, :].astype(int).tolist()
            heights = Info['heights'][0, :].astype(int).tolist()

    # 建立資料集
        # 建立 Dataset (已改寫成抽幀方式)
        dataset = VideoDataset(
            videos_dir=videos_dir,
            video_names=video_names,
            scores=scores,
            widths=widths,
            heights=heights
        )


    # 檢查資料集長度
    print(f"Length of dataset: {len(dataset)}")

    for i in range(len(dataset)):
        print("[",i,"/",len(dataset),"]current video names:", video_names[i])
        width, height = widths[i], heights[i]
        print(f"Resolution: {width}x{height}")


        current_data = dataset[i]

        current_video = current_data['video']  # current_video shape should be [N, C, H, W]
        current_score = current_data['score']
        print(f'Processing video {i}: length {current_video.shape[0]}')
        print('Current video: ', current_video.shape)
        # final3D_features = exract3D(current_video, device).to(device)
        try:
            final3D_features = exract3D(current_video, device).to(device)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ GPU 記憶體不足，將嘗試以低解析度重試影片 {video_names[i]}")
                torch.cuda.empty_cache()

                # 降解析度後重新呼叫 exract3D（這裡傳入 resized 版本）
                downscaled_video = F.interpolate(current_video, size=(1080 , 1920), mode='bilinear', align_corners=False)
                try:
                    final3D_features = exract3D(downscaled_video, device).to(device)
                except Exception as ee:
                    print(f"⛔ 降解析度後仍失敗，跳過影片 {video_names[i]}：{ee}")
                    continue  # 跳過這個影片
            else:
                raise e
        if torch.isnan(final3D_features).any() or torch.isinf(final3D_features).any():
            print("NaN or Inf detected in final3D_features.")
        weighted_frame_features, all_indices,diffs = get_additional_features(current_video, device)
        if torch.isnan(weighted_frame_features).any() or torch.isinf(weighted_frame_features).any():
            print("NaN or Inf detected in weighted_frame_features.")
        weights = get_weights(diffs)
        all_weights = weights[all_indices]

        repeated_3D = final3D_features.repeat(weighted_frame_features.size(0), 1)
        if torch.isnan(repeated_3D).any() or torch.isinf(repeated_3D).any():
            print("NaN or Inf detected in repeated_3D1.")
        repeated_3D = repeated_3D.to(weighted_frame_features.device)
        if torch.isnan(repeated_3D).any() or torch.isinf(repeated_3D).any():
            print("NaN or Inf detected in repeated_3D2.")
        print(f'repeated_3D shape: {repeated_3D.shape}')
        final_features = torch.cat((weighted_frame_features, repeated_3D), dim=1)
        print(f'final_features shape: {final_features.shape}')
        weight_distribution_df = pd.DataFrame({
            'Frame Index': all_indices.cpu().numpy(),
            'Weight': all_weights.cpu().numpy(),
        })

        print(weight_distribution_df)
        # output_file = os.path.join(features_dir, f'{i}SCCOMEweight.csv')
        # weight_distribution_df.to_csv(output_file, index=False)

        if torch.isnan(final_features).any() or torch.isinf(final_features).any():
            print("NaN or Inf detected in final_features.")
        np.save(os.path.join(features_dir, f'{i}SCCOME.npy'),
                final_features.detach().cpu().numpy())
        np.save(os.path.join(features_dir, f'{i}_score.npy'), current_score)

