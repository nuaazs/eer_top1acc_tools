@@ -0,0 +1,69 @@
# coding = utf-8
# @Time    : 2023-05-14  22:17:20
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: 利用后端API接口，传入VAD后数据文件，获取对应文件的embedding特征，保存为numpy文件.
import requests
from tqdm import tqdm
import numpy as np
import os
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--fold_path', type=str, default='/datasets_hdd/datasets/cjsd_vad_0.1_0.1/',help='After vad data path')
parser.add_argument('--dst_path', type=str, default="/datasets_hdd/datasets/cjsd_0101_embeddings_ecapatdnn_16k",help='Path for output embedding npy files')
parser.add_argument('--thread', type=int, default=32,help='Thread number, same as the number of API server')
parser.add_argument('--url', type=str, default="http://127.0.0.1:8888/get_embedding/file",help='API server url')
args = parser.parse_args()

def get_embedding(file_path,savepath=args.dst_path):
    """获取该文件的embedding特征
    Args:
        file_path (str): 文件路径
    Returns:
        None
    """
    filename = file_path.split('/')[-1].split('.')[0]
    payload={"spkid":str(filename)}
    files=[
    ('wav_file',(file_path.split('/')[-1],open(file_path,'rb'),'application/octet-stream'))
    ]
    try:
        response = requests.request("POST", args.url,data=payload, files=files)
        if "embeddings" not in response.json():
            print("!!!!!!!!Error!!!!!!!!"*2)
            print(response.json())
            print("!!!!!!!!Error!!!!!!!!"*2)
            return 0
        else:
            emb = np.array(response.json()["embeddings"][0]) # shape (len_of_emb,)
            if savepath:
                output_path = os.path.join(savepath,filename)
                np.save(output_path,emb)
    except Exception as e:
        print("!!!!!!!!Error!!!!!!!!"*2)
        print(e)
        print("!!!!!!!!Error!!!!!!!!"*2)
        return 0
    return 1

if __name__ == "__main__":
    # make dst folder
    os.makedirs(args.dst_path,exist_ok=True)

    # get all wavs in args.fold_path, recursive
    all_wavs = []
    for file in os.listdir(args.fold_path):
        if file.endswith(".wav"):
            print(os.path.join(args.fold_path, file))
            all_wavs.append(os.path.join(args.fold_path, file))

    # multi process call get_embedding
    from multiprocessing import Pool
    pool = Pool(processes=args.thread)
    embeddings_list = list(tqdm(pool.imap(get_embedding, all_wavs), total=len(all_wavs)))
    pool.close()
    pool.join()
print("Done!")
print(f"* Total accecpt #{len(all_wavs)} files, success #{len([_r for _r in embeddings_list if _r>0])}")