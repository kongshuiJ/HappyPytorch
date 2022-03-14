import os
import shutil
import random


def add_training__list(data_type, dir_path, output_file_path):
    # 获取文件名
    file_names = os.listdir(dir_path)
    # 文件名拼接路径
    file_list = [file for file in file_names]
    print(file_list)

    for file in file_names:
        src = "%s/%s" % (data_type, file)
        os.system("echo %s >> %s" % (src, output_file_path))


def copy_audio_files(src_dir_path, target_dir_path):
    os.system("cp -rf %s/* %s" % (src_dir_path, target_dir_path))


test_audio_dir_path = './AudioData'
training_file_path = "%s/training_list.txt" % test_audio_dir_path
testing_file_path = "%s/testing_list.txt" % test_audio_dir_path

os.system("rm %s" % training_file_path)
os.system("rm %s" % testing_file_path)

# training__list
add_training__list('brick', './audio/train/brick_output/', training_file_path)
add_training__list('floor', './audio/train/floor_output/', training_file_path)
add_training__list('carpet', './audio/train/carpet_output/', training_file_path)

copy_audio_files('./audio/train/brick_output/', "%s/brick/" % test_audio_dir_path)
copy_audio_files('./audio/train/floor_output/', "%s/floor/" % test_audio_dir_path)
copy_audio_files('./audio/train/carpet_output/', "%s/carpet/" % test_audio_dir_path)

# # testing_list
add_training__list('brick', './audio/test/brick_output/', testing_file_path)
add_training__list('floor', './audio/test/floor_output/', testing_file_path)
add_training__list('carpet', './audio/test/carpet_output/', testing_file_path)

copy_audio_files('./audio/test/brick_output/', "%s/brick/" % test_audio_dir_path)
copy_audio_files('./audio/test/floor_output/', "%s/floor/" % test_audio_dir_path)
copy_audio_files('./audio/test/carpet_output/', "%s/carpet/" % test_audio_dir_path)
