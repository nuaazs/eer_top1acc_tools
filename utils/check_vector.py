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

if __name__ == '__main__':
    # /datasets_hdd/datasets/cjsd_vad
    # 读取二进制文件
    data_ = np.fromfile("/home/zhaosheng/get_cjsd_embeddings/vector.bin", dtype=np.float32)
    data_ = data_.reshape(-1, 192)
    print(f"Raw Data shape: {data_.shape}")
