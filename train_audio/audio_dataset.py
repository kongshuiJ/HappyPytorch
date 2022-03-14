import os
import fnmatch
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):
    def __init__(self, data_folder, sr=16000, dimension=8192):
        self.data_folder = data_folder
        self.sr = sr
        self.dim = dimension

        # 获取音频名列表
        self.wav_list = []
        for root, dirnames, filenames in os.walk(data_folder):
            for filename in fnmatch.filter(filenames, "*.wav"):  # 实现列表特殊字符的过滤或筛选,返回符合匹配“.wav”字符列表
                self.wav_list.append(os.path.join(root, filename))

    def __getitem__(self, item):
        # 读取一个音频文件，返回每个音频数据
        filename = self.wav_list[item]
        wb_wav, _ = librosa.load(filename, sr=self.sr)

        # 取 帧
        if len(wb_wav) >= self.dim:
            max_audio_start = len(wb_wav) - self.dim
            audio_start = np.random.randint(0, max_audio_start)
            wb_wav = wb_wav[audio_start: audio_start + self.dim]
        else:
            wb_wav = np.pad(wb_wav, (0, self.dim - len(wb_wav)), "constant")

        return wb_wav, filename

    def __len__(self):
        # 音频文件的总数
        return len(self.wav_list)


import os
import fnmatch
import numpy as np
import librosa
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, data_folder, sr=16000, dimension=8192):
        self.data_folder = data_folder
        self.sr = sr
        self.dim = dimension

        # 获取音频名列表
        self.wav_list = []
        for root, dirnames, filenames in os.walk(data_folder):
            for filename in fnmatch.filter(filenames, "*.wav"):  # 实现列表特殊字符的过滤或筛选,返回符合匹配“.wav”字符列表
                self.wav_list.append(os.path.join(root, filename))

    def __getitem__(self, item):
        # 读取一个音频文件，返回每个音频数据
        filename = self.wav_list[item]
        wb_wav, _ = librosa.load(filename, sr=self.sr)
        label = 0
        # 取 帧
        if len(wb_wav) >= self.dim:
            max_audio_start = len(wb_wav) - self.dim
            audio_start = np.random.randint(0, max_audio_start)
            wb_wav = wb_wav[audio_start: audio_start + self.dim]
            label = 1 if 'brick' in filename.split('/')[-1] else 0
        else:
            wb_wav = np.pad(wb_wav, (0, self.dim - len(wb_wav)), "constant")

        return wb_wav, filename, label

    def __len__(self):
        # 音频文件的总数
        return len(self.wav_list)


# 实例化AudioDataset对象
train_set = AudioDataset("./data", sr=16000)

for data in train_set:
    print(data)

# for i, data in enumerate(train_set):
#     wb_wav, filname = data
#     print(i, wb_wav.shape, filname)
#
#     if i == 3:
#         break
#     # 0 (8192,) ./data\p225_001.wav
#     # 1 (8192,) ./data\p225_002.wav
#     # 2 (8192,) ./data\p225_003.wav
#     # 3 (8192,) ./data\p225_004.wav
