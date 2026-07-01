"""
Shin2017A 数据集预处理脚本 (Braindecode)
=========================================
本脚本展示了如何使用 Braindecode 对运动想象 EEG 数据进行完整的预处理流程。
包括：通道选择、重采样、带通滤波、窗口切分、指数滑动标准化与均值方差归一化。
"""

import os

DEFAULT_MNE_DATA_PATH = '/home/xuyuanjun/leeg/data/neural/moabb/Shin2017OpenA/download/'

os.makedirs(DEFAULT_MNE_DATA_PATH, exist_ok=True)
os.environ['MNE_DATA'] = DEFAULT_MNE_DATA_PATH
os.environ['MNE_DATASETS_BBCIFNIRS_PATH'] = DEFAULT_MNE_DATA_PATH
os.environ['MNE_DATASETS_BBCIFNIRS'] = DEFAULT_MNE_DATA_PATH

print(f"[初始化] MNE_DATA 设置为: {DEFAULT_MNE_DATA_PATH}")


# ------------------------------
# 1. 通过 MOABB 加载数据集
# ------------------------------
# 指定数据集名称、被试编号（示例中用 1、2 号），accept=True 表示同意数据使用协议
from moabb.datasets import Shin2017A
dataset = Shin2017A(accept='True')
 
from braindecode.datasets import MOABBDataset

# ------------------------------
# 1. 通过 MOABB 加载数据集
# ------------------------------
# 指定数据集名称、被试编号（示例中用 1、2 号），accept=True 表示同意数据使用协议
dataset = MOABBDataset(
    dataset_name='Shin2017A',      # MOABB 中的数据集标识符
)   


import numpy as np
from braindecode.preprocessing import (
    preprocess,
    create_windows_from_events,
)
from braindecode.preprocessing import Pick, Resample, Filter

# ------------------------------
# 2. 定义原始数据预处理步骤
# ------------------------------
preprocessors = [
    # 2.1 仅保留 EEG 通道（丢弃 NIRS、EOG 等）
    Pick(picks='eeg'),             # 等效于选择 data_chs 类型为 'eeg'
    Filter(l_freq=0.5, h_freq=99.5)
]

# # 应用上述预处理操作（原地修改 dataset 中的 raw 对象）
preprocess(dataset, preprocessors)

windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=400,    # set [-500, 500] to avoid duration = 0
    trial_stop_offset_samples=0,  
    window_size_samples=256,          # freq * time
    window_stride_samples=256,
    drop_last_window=True,
    preload=True,
    use_mne_epochs=True,
)

print(f"预处理后的窗口总数: {len(windows_dataset)}")
# 查看第一个窗口的数据形状和标签
X, y, window_metadata = windows_dataset[0]
print(f"单个样本数据形状: {X.shape}")   # (通道数, 时间点)
print(f"对应标签: {y}")

first_epochs = windows_dataset.datasets[0].windows
info = first_epochs.info
ch_names = np.array([info['chs'][i]['ch_name'] for i in range(len(info['chs']))])
print(ch_names)

windows_dataset.save(
    path=f'/home/xuyuanjun/leeg/data/neural/moabb/Shin2017OpenA/download/EEG_process',
    overwrite=True,
)


