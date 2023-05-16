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
from tqdm import tqdm
import random
# set random seed
random.seed(1997)
parser = argparse.ArgumentParser()
parser.add_argument('--fold_path', type=str, default="/datasets_hdd/datasets/cjsd_embeddings_ecapatdnn_16k", help='Folder for embedding npy files')
parser.add_argument('--save_path', type=str, default='./vector.bin', help='vector bin save path')
parser.add_argument('--save_txt_path', type=str, default='./vector.txt', help='vector txt save path')
parser.add_argument('--save_tiny_folder', type=str, default='./data/temp', help='vector txt save path')
parser.add_argument('--thread', type=int, default=1, help='')
# parser.add_argument('--save_txt_path', type=str, default='./vector.txt', help='vector txt save path')

args = parser.parse_args()

def filelist_to_dict(filelist):
    process_id = os.getpid()+random.randint(0,10000)
    data = {}
    data_list = []
    id_list = []
    for file in tqdm(filelist):
        file_name = file.split('/')[-1].split('.')[0]
        data[file_name] = np.load(os.path.join(args.fold_path, file))
    for key in data.keys():
        data_list.append(data[key])
        id_list.append(key)
        # 将data_list用二进制的形式保存到vectorDB.bin
    data_ = np.array(data_list, dtype=np.float32)
    data_.tofile(os.path.join(args.save_tiny_folder, f"vector_{process_id}.bin"))
        
    with open(os.path.join(args.save_tiny_folder, f"vector_{process_id}.txt"), 'w') as f:
        for id in id_list:
            f.write(id + '\n')
    print(f"Raw Data shape: {data_.shape}")
    path_pair = (os.path.join(args.save_tiny_folder, f"vector_{process_id}.bin"), os.path.join(args.save_tiny_folder, f"vector_{process_id}.txt"))
    return path_pair

if __name__ == '__main__':
    # if save_tiny_folder not exist, mkdir
    os.makedirs(args.save_tiny_folder, exist_ok=True)
    # /datasets_hdd/datasets/cjsd_vad
    # read all npy file from args.fold_path, recursive
    # add to data, key is filename, value is npy data
    data = {}
    npy_files = sorted([_file for _file in os.listdir(args.fold_path) if _file.endswith(".npy")])

    if args.thread <= 1:
        path_pair = filelist_to_dict(npy_files)
        print(f"Saved data to {path_pair[0]}")
        print(f"Saved id to {path_pair[1]}")
    else:
        # split npy_file to #thread parts
        npy_files = np.array_split(npy_files, args.thread)
        # multi process call filelist_to_dict
        import multiprocessing
        pool = multiprocessing.Pool(processes=args.thread)
        pair_list = pool.map(filelist_to_dict, npy_files)
        pool.close()
        pool.join()
        print(pair_list)
        print(f"Saved data to {args.save_path}")