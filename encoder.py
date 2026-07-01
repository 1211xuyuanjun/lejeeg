"""
多核 Conv1D 频率编码器 + 时频图编码器 (3层精简版)
"""

import torch
import torch.nn as nn
from einops import rearrange


def calc_conv_output_size(input_size, kernel_size, stride, padding):
    """计算 Conv1d/Conv2d 单层的输出尺寸"""
    return (input_size + 2 * padding - kernel_size) // stride + 1


class MultiKernelConvEncoder(nn.Module):
    """
    多核 Conv1D 频率编码器
    每个分支固定3层 Conv1d:
        Layer 1: kernel=k,  stride=1,  提取宽频率特征
        Layer 2: kernel=3,  stride=1,  精炼
        Layer 3: kernel=3,  stride=stride, 下采样
    
    输入: (b, c, t)
    输出: (b, c, t_1, out_dim)
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        kernel_sizes: list = None,
        out_dim: int = None,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 7, 15, 31]
        if out_dim is None:
            out_dim = hidden_dim

        self.hidden_dim = hidden_dim
        self.kernel_sizes = kernel_sizes
        self.num_branches = len(kernel_sizes)

        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            branch = nn.Sequential(
                # Layer 1: 大 kernel 提取频率
                nn.Conv1d(1, hidden_dim, kernel_size=k, stride=4, padding=k // 2),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                # Layer 2: 小 kernel 精炼
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                # Layer 3: 下采样
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            )
            self.branches.append(branch)

        self.conv_out = nn.Linear(self.num_branches * hidden_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        x = x.reshape(b * c, 1, t)

        branch_outputs = []
        for branch in self.branches:
            out = branch(x)  # (b*c, h, t_1)
            out = out.reshape(b, c, self.hidden_dim, -1)  # (b, c, h, t_1)
            branch_outputs.append(out)

        # (b, c, f, h, t_1) -> (b, c, t_1, f*h)
        result = torch.stack(branch_outputs, dim=2)
        result = rearrange(result, 'b c f h t -> b c t (f h)')
        result = self.conv_out(result)
        return result


class SpectrogramEncoder(nn.Module):
    """
    时频图编码器 (固定3层 Conv2d)
    
    默认配置:
        Layer 1: kernel=(3,7), stride=(1,2)  频率保持, 时间减半
        Layer 2: kernel=(3,3), stride=(2,2)   频率减半, 时间减半  
        Layer 3: kernel=(3,3), stride=(2,2)   频率减半, 时间减半
    
    输入: (b, c, f, t)
    输出: (b, c, t_1, out_dim)
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        out_dim: int = None,
        freq_bins: int = None,
        time_steps: int = None,
    ):
        super().__init__()
        if out_dim is None:
            out_dim = hidden_dim

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        self.kernel_sizes = ((3, 3), (3, 3), (3, 3))
        self.strides = ((2, 1), (2, 1), (2, 1))

        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, hidden_dim, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            # Layer 2
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            # Layer 3
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        # 计算输出尺寸
        if freq_bins is not None and time_steps is not None:
            f, t = freq_bins, time_steps
            for k, s in zip(self.kernel_sizes, self.strides):
                pad = (k[0] // 2, k[1] // 2)
                f = calc_conv_output_size(f, k[0], s[0], pad[0])
                t = calc_conv_output_size(t, k[1], s[1], pad[1])
            self.f_1 = f
            self.t_1 = t
            self.conv_out = nn.Linear(hidden_dim * f, out_dim, bias=False)
        else:
            self.f_1 = None
            self.t_1 = None
            self.conv_out = nn.LazyLinear(out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, f, t = x.shape
        x = x.reshape(b * c, 1, f, t)
        x = self.conv(x)  # (b*c, h, f_1, t_1)
        x = rearrange(x, '(b c) h f_1 t_1 -> b c t_1 (h f_1)', b=b)
        x = self.conv_out(x)
        return x

    def get_output_size(self, freq_bins: int, time_steps: int):
        f, t = freq_bins, time_steps
        for k, s in zip(self.kernel_sizes, self.strides):
            pad = (k[0] // 2, k[1] // 2)
            f = calc_conv_output_size(f, k[0], s[0], pad[0])
            t = calc_conv_output_size(t, k[1], s[1], pad[1])
        return f, t



if __name__ == "__main__":
    # 原始 EEG
    eeg = torch.randn(8, 32, 1024)
    b, c, t = eeg.shape

    # 方案 A：直接使用 MultiKernelConvEncoder
    encoder_multi = MultiKernelConvEncoder(out_dim=128)
    feat_multi = encoder_multi(eeg)   # (8,32,33,128)
    print(feat_multi.shape)

    # 方案 B：先 STFT，再用 SpectrogramEncoder（修正后的配置）
    x = rearrange(eeg, 'b c t -> (b c) t')
    spec = torch.stft(
        x,
        n_fft=127,           # 128点 FFT
        hop_length=32,       # 32点帧移
        win_length=127,
        window=torch.hann_window(127).to(x.device),
        center=True,
        onesided=True,
        return_complex=True
    )  # shape: (b*c, n_fft//2+1, time_frames)

    encoder_spec = SpectrogramEncoder(out_dim=128)
    spec = rearrange(spec, '(b c) f t -> b c f t', b=b)
    magnitude = spec.abs() # (freq_bins, time_frames)
    magnitude_log = 20 * torch.log10(magnitude + 1e-10)         
    feat_spec = encoder_spec(magnitude_log)
    print(feat_spec.shape)
