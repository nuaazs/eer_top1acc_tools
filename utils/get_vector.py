# coding = utf-8
# @Time    : 2023-05-14  22:32:54
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: export redis data to a python dict and save as a npy file.
# # embedding目录 --> vector.bin  id.txt
# vector.bin是shape为(音频数量*特征长度)的二进制文件
# id.txt 为ID列表，与bin文件顺序一一对应。

import numpy as np
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--fold_path', type=str, default="/datasets_hdd/datasets/cjsd_embeddings_ecapatdnn_16k", help='Folder for embedding npy files')
parser.add_argument('--save_path', type=str, default='./vector.bin', help='vector bin save path')
parser.add_argument('--save_txt_path', type=str, default='./vector.txt', help='vector txt save path')
args = parser.parse_args()

if __name__ == '__main__':
    # /datasets_hdd/datasets/cjsd_vad
    # read all npy file from args.fold_path, recursive
    # add to data, key is filename, value is npy data
    data = {}
    for root, dirs, files in os.walk(args.fold_path):
        for file in files:
            if file.endswith(".npy"):
                file_name = file.split('.')[0]
                data[file_name] = np.load(os.path.join(root, file))

    data_list = []
    id_list = []
    for key in data.keys():
        data_list.append(data[key])
        id_list.append(key)

    # 将data_list用二进制的形式保存到vectorDB.bin
    data_ = np.array(data_list, dtype=np.float32)
    print(f"Data shape: {data_.shape}")
    data_.tofile(args.save_path)
    print(f"Saved data to {args.save_path}")

    with open(args.save_txt_path, 'w') as f:
        for id in id_list:
            f.write(id + '\n')
    print(f"Saved id to {args.save_txt_path}")
    # 读取二进制文件
    data_ = np.fromfile(args.save_path, dtype=np.float32)
    print(f"Raw Data shape: {data_.shape}")
