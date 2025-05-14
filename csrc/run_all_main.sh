#!/bin/bash
# 定义 M 数组
M=(256 512 1024 2048 4096 8192 )
# 定义 SK 数组，与 M 数组对应
SK=(4 4 4 4 4 4 )

# 使用索引遍历 M 和 SK 数组
for i in "${!M[@]}"; do
    m=${M[$i]}    # 从 M 数组获取当前 m 值
    k=$m          # K 与 M 相同
    n=$m          # N 与 M 相同
    s=50          # 固定稀疏度
    sk=${SK[$i]}  # 从 SK 数组获取对应的 sk 值
    
    echo "Running spinfer and cublas test case: M=$m, K=$k, N=$n, S=$s, SK=$sk"
    ./spmm_test $m $k $n $s $sk
done