import os
from pydub import AudioSegment
import numpy as np


def segment_wav(input_dir_path, output_dir_path):
    for file in os.listdir(input_dir_path):  # 遍历文件
        path1 = input_dir_path + '/' + file
        filename = file.split('.')[0]  # 不带 .wav的文件名
        ########################处理音频文件#######################
        audio = AudioSegment.from_file(path1, "wav")
        audio_time = len(audio)  # 获取待切割音频的时长，单位是毫秒
        print('audio_time:', audio_time)
        cut_parameters = np.arange(1, audio_time / 1000, 5)  # np.arange()函数第一个参数为起点，第二个参数为终点，第三个参数为步长（1秒）
        start_time = int(0)  # 开始时间设为0
        ########################根据数组切割音频####################
        for t in cut_parameters:
            stop_time = int(t * 1000)  # pydub以毫秒为单位工作
            # print(stop_time)
            audio_chunk = audio[start_time:stop_time]  # 音频切割按开始时间到结束时间切割
            print("split at [{}:{}] ms".format(start_time, stop_time))
            audio_chunk.export(output_dir_path + '/' + filename + "-{}.wav".format(int(t / 1)), format="wav")  # 保存音频文件
            start_time = stop_time - 400  # 开始时间变为结束时间前400ms---------也就是叠加上一段音频末尾的400ms
            print('finish')


# train
segment_wav('./audio/train/floor_input/', './audio/train/floor_output/')
segment_wav('./audio/train/carpet_input/', './audio/train/carpet_output/')
segment_wav('./audio/train/brick_input/', './audio/train/brick_output/')

# test
segment_wav('./audio/test/floor_input/', './audio/test/floor_output/')
segment_wav('./audio/test/carpet_input/', './audio/test/carpet_output/')
segment_wav('./audio/test/brick_input/', './audio/test/brick_output/')