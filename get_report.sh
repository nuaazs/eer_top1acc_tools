#!/bin/bash
# 给定两批数据，数据A:所有底库文件，数据B:所有测试文件

EMB_SIZE=192
api_thread=1
calc_thread=96
# 指定目录地址
a_dir=/home/zhaoxt20/data/voxceleb1/feat
b_dir=/home/zhaoxt20/data/voxceleb1/feat
a_vad_dir=/home/zhaoxt20/data/voxceleb1/vad
b_vad_dir=/home/zhaoxt20/data/voxceleb1/vad
# 指定输出地址
a_emb_dir=/home/zhaoxt20/data/voxceleb1/emb
b_emb_dir=/home/zhaoxt20/data/voxceleb1/emb

a_bin_path=/home/zhaosheng/get_cjsd_embeddings/vector.bin
a_txt_path=/home/zhaosheng/get_cjsd_embeddings/vector.txt
b_bin_path=/home/zhaosheng/get_cjsd_embeddings/vector.bin
b_txt_path=/home/zhaosheng/get_cjsd_embeddings/vector.txt

a_split_dir=/home/zhaosheng/get_cjsd_embeddings/a_split
b_split_dir=/home/zhaosheng/get_cjsd_embeddings/b_split

rm -rf $a_split_dir
rm -rf $b_split_dir
# # 1. 对数据A进行预处理
# python get_vad.py --fold_path $a_dir --dst_path $a_vad_dir --thread $api_thread
# # 2. 对数据B进行预处理
# python get_vad.py --fold_path $b_dir --dst_path $b_vad_dir --thread $api_thread
# # 3. 对预处理后的A进行特征提取
# python get_embedding.py --fold_path $a_vad_dir --dst_path $a_emb_dir --thread $api_thread
# # 4. 对预处理后的B进行特征提取
# python get_embedding.py --fold_path $b_vad_dir --dst_path $b_emb_dir --thread $api_thread
# # 5. 获取a.bin a.txt 和 b.bin b.txt
# python get_vector.py --fold_path $a_emb_dir --save_path $a_bin_path --save_txt_path $a_txt_path
# python get_vector.py --fold_path $b_emb_dir --save_path $b_bin_path --save_txt_path $b_txt_path

# 6. split a.bin a.txt为多个小文件,格式 1.bin 1.txt ...
python split_vector.py --raw_bin_path $a_bin_path --raw_txt_path $a_txt_path --number $calc_thread --save_folder $a_split_dir

b_len=$(cat $b_txt_path | wc -l)

# 7. 计算分数
for file_num in $(seq 0 $((calc_thread-1)))
do
    echo "file_num: $file_num"
    # 获取txt文件和bin文件地址
    txt_path=$a_split_dir/id_$file_num.txt
    bin_path=$a_split_dir/vector_$file_num.bin
    # a的长度为txt文件的行数
    a_len=$(cat $txt_path | wc -l)
    # Usage: program_name NUM_CJSD NUM_BLACK EMB_SIZE DB1 DB2 ID1 ID2 OUTPUT_PATH
    ./top1 $a_len $b_len $EMB_SIZE $bin_path $b_bin_path $txt_path $b_txt_path $a_split_dir/$file_num.score & #> /dev/null  # $a_split_dir/$file_num.score
done
# 等待所有进程结束
wait

echo "all process done. Now calc top1 acc"
for file_num in $(seq 0 $((calc_thread-1)))
do
    echo "file_num: $file_num"
    # 获取txt文件和bin文件地址
    txt_path=${a_split_dir}/id_$file_num.txt
    bin_path=${a_split_dir}/vector_$file_num.bin
    # a的长度为txt文件的行数
    a_len=$(cat $txt_path | wc -l)
    # Usage: program_name NUM_CJSD NUM_BLACK EMB_SIZE DB1 DB2 ID1 ID2 OUTPUT_PATH
    # echo "./top1acc ${a_split_dir}/${file_num}.score 0.1 0.9 0.05 ${a_split_dir}/${file_num}_results"
    ./top1acc ${a_split_dir}/${file_num}.score 0.1 0.9 0.05 ${a_split_dir}/${file_num}_results & #> /dev/null  # $a_split_dir/$file_num.score
done
wait
echo "Done"

# Merge all results
python merge_top1_acc_result.py --root_path a_split_dir --save_path ./a_split.csv







# # rm all sub bin and id
# rm $a_split_dir/*_*.bin
# rm $a_split_dir/*_*.txt
# rm $b_split_dir/*.log

# # # 9. 计算eer
# for file_num in $(seq 0 $((calc_thread-1)))
# do
#     echo "file_num: $file_num"
#     # 获取eer结果
#     # ./eer --input $a_split_dir/$file_num.score --start-threshold 0.1 --end-threshold 1.0 --step 0.3 --plot 0 --savepath $a_split_dir/${file_num}_eer_results &
#     echo "./eer --input $a_split_dir/$file_num.score --start-threshold 0.1 --end-threshold 1.0 --step 0.3 --plot 0 --savepath $a_split_dir/${file_num}_eer_results"
# done


# ./eer --start-threshold 0.1 --end-threshold 1.0 --step 0.3 --plot 0 --savepath $a_split_dir/eer_results