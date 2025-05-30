#!/bin/bash
M=(8192 4096 32000 32000 28672 5120 5120  3584  4096 13824 8192  18944 14336 4096 8192  11008 32000 20480 3584 21504 7168 28672 7168 27648 9216 36864 9216 36864 12288 49152 12288)
K=(29568 4096 5120 8192  8192  5120 13824 20480  11008 5120 8192 3584  4096  14336 28672 4096 4096 3584 18944 7168 7168 7168 28672 9216 9216 9216 36864 12288 12288 12288 49152)
N=(8 16 32)
SPLIT_K=(1)
SPARSITY=(40 50 60 70)
if [ ${#M[@]} -ne ${#K[@]} ]; then
    echo "Error: M and K arrays must have the same length."
    exit 1
fi
for ((i=0; i<${#M[@]}; i++)); do
    m=${M[i]}
    k=${K[i]}
    for n in "${N[@]}"; do
        for s in "${SPARSITY[@]}"; do
            for sk in "${SPLIT_K[@]}"; do
                echo "Running sputnik test case: M=$m, K=$k, N=$n, S=$s, SK=$sk"
                ./spmm_test_sputnik $m $k $n $s $sk
            done
        done
    done
done