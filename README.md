## Top 1 准确率计算工具 😎

输入文件：
1. 目录A：`dir_a`
2. 目录B: `dir_b`

输出结果：
1. 各阈值下ACC, Recall, Precision等结果csv.

## 使用方法 🚀
### 步骤一：文件准备 📝
1. 目录A `dir_a`: 待测试数据，例如293万待测试音频+100个埋点数据。
2. 目录B `dir_b`：底库数据，例如8万黑库音频文件。
3. 每个文件的文件名为`<spkid>_<xxxx>.wav`

### 步骤二：声纹编码 🎙️
利用声纹服务API端口，将步骤一中的目录A及目录B中的所有文件进行编码，每个音频文件生成一个独一的特征文件，用numpy格式保存。每个numpy的shape为`(n,)`，其中`n`为特征的长度。
获得：
1. 声纹特征目录A `emb_a`
2. 声纹特征目录B `emb_b`
3. 每个文件的文件名为`<spkid>_<xxxx>.npy`


### 步骤三：生成二进制文件 💻
1. 将声纹特征目录A `emb_a`中的所有特征进行堆叠，生成shape为`(N,n)`的二维数组，其中`n`为特征的长度，`N`为音频个数。
2. 将shape为`(N,n)`的二维数组利用二进制保存，格式为`float32`，生成`vector_a.bin`。同时生成`vector_a.txt`。`vector_a.bin`包含了所有声纹信息。`vector_b.txt`按顺序列出了所有声纹特征的说话人ID。
3. 对于声纹特征目录B `emb_b` 进行同样的操作获得`vector_b.txt`和`vector_b.bin`。


### 步骤四（可选）：加入埋点数据 🔎


### 步骤五：文件分割 📑
由于所有待测文件，可将原始`vector_a.bin`进行分割，以便于进行后续并行碰撞。
例如将其分割为64份，并保存在`vector_a_data`目录下：
```shell
python split_vector.py --raw_bin_path vector_a.bin --raw_txt_path vector_a.txt --number 64 --save_folder vector_a_data
```


### 步骤六：计算余弦相似度 🧮
利用可执行程序`top1`计算得分，输出两个bin的碰撞结果。
1. `vector_a_1.bin`:`vector_a.bin`分割后某个子集。
2. `a_len`:`vector_a_1.bin`的特征数量。
3. `b_len`:黑库特征数量。
4. `EMB_SIZE`:单一声纹特征的长度。
5. `b_bin_path`:声纹黑库特征文件，即`vector_b.bin`
6. `txt_path`:`vector_a_1.bin`对应的说话人ID。
7. `b_txt_path`:`vector_b.bin`对应的说话人ID。
8. `$a_split_dir/$file_num.score`:结果的保存目录

使用示例：
```shell
# Usage: program_name NUM_CJSD NUM_BLACK EMB_SIZE DB1 DB2 ID1 ID2 OUTPUT_PATH
./top1 $a_len $b_len $EMB_SIZE $bin_path $b_bin_path $txt_path $b_txt_path $a_split_dir/$file_num.score &
```
计算所有子集的碰撞得分：
```shell
b_len=$(cat $b_txt_path | wc -l)

# 7. 计算分数
b_txt_path="vector_b.txt"
calc_thread=64 # 64个子集
a_split_dir="vector_a_data"
b_len=$(cat $b_txt_path | wc -l)
for file_num in $(seq 0 $((calc_thread-1)))
do
    echo "file_num: $file_num"
    # 获取txt文件和bin文件地址
    txt_path=$a_split_dir/id_$file_num.txt
    bin_path=$a_split_dir/vector_$file_num.bin
    # a的长度为txt文件的行数
    a_len=$(cat $txt_path | wc -l)
    ./top1 $a_len $b_len $EMB_SIZE $bin_path $b_bin_path $txt_path $b_txt_path $a_split_dir/$file_num.score &
done
wait
```

### 步骤七：统计TP/TN/FP/FN 📊
对于步骤六中的每个结果，利用可制成程序`top1acc`分别统计不同阈值下的TP/TN/FP/FN等信息
1. `score_file_path`:步骤六中输出的score文件即`$a_split_dir/$file_num.score`
2. `th_start`:阈值起始值
3. `th_stop`:阈值中止值
4. `th_step`:阈值遍历的步长
5. `save_dir`:结果保存的路径

```shell
Usage: ./top1acc score_file_path th_start th_stop th_step save_dir
```

```shell
echo "all process done. Now calc top1 acc"
for file_num in $(seq 0 $((calc_thread-1)))
do
    ./top1acc ${a_split_dir}/${file_num}.score 0.1 0.9 0.05 ${a_split_dir}/${file_num}_results &
done
wait
echo "Done"
```

### 步骤八：合并所有子集结果 🤝
步骤一至步骤七获得了每个`vector_a.bin`的子集与`vector_b.bin`的测试结果。
最后利用`merge_top1_acc_result.py`对结果进行合并，获得`vector_a.bin`与`vector_b.bin`的完整测试结果。
```shell
python merge_top1_acc_result.py --root_path $a_split_dir --save_path $a_csv_path
```
