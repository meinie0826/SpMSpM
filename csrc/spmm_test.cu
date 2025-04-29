/***************************************************************************
 * Copyright 2025 The SpInfer Authors. All rights reserved.
 * Copyright 2023 The FLash-LLM Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/
#include "./SpMM_API.cuh"

#include "./spmm_test_utils.h"
#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <stdio.h>
#define TTTTT 0
int main(int argc, char **argv) {

    // int M_GLOBAL = 8192;
    // int K_GLOBAL = 8192;
    // int N_GLOBAL = 1024;

    int M_GLOBAL = 36864;
    int K_GLOBAL = 36864;
    int N_GLOBAL = 36864;
    int MATRIX_A_PRUNING_PERCENTAGE = 50;
    int SPLIT_K = 1;
    cublasStatus_t cublas_status;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Host memory
    half *A_h = NULL; // row major
    half *B_h = NULL; // col major
    // Device memory
    half *A = NULL;
    half *B = NULL;
    A_h = (half *)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
    B_h = (half *)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
   
    if (A_h == NULL || B_h == NULL) {
        printf("Error in CPU Malloc!\n");
        exit(-1);
    }
    cudaMalloc(reinterpret_cast<void **>(&A), sizeof(half) * M_GLOBAL * K_GLOBAL);
    cudaMalloc(reinterpret_cast<void **>(&B), sizeof(half) * N_GLOBAL * K_GLOBAL);

    checkLastCudaError(__LINE__);
    if (A == NULL || B == NULL) {
        printf("Error in cudaMalloc!\n");
        exit(-1);
    }
    //

    init_host_matrices(A_h, B_h, M_GLOBAL, K_GLOBAL, N_GLOBAL, MATRIX_A_PRUNING_PERCENTAGE);

    printf("Preparing dense data for GPU...\n");
    cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    checkLastCudaError(__LINE__);

    // CUBLAS
    /////////////////////////////////////////////////////////////////////////////////////////////////
    printf("Launching CuBlas...\n");
    half *D_cublas = NULL;
    cudaMalloc(reinterpret_cast<void **>(&D_cublas), sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (D_cublas == NULL) {
        printf("Error in spmm_test.cu: line %d cudaMalloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemset(D_cublas, 0, sizeof(half) * M_GLOBAL * N_GLOBAL);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, 0);

    int m = M_GLOBAL, n = N_GLOBAL, k = K_GLOBAL;
    const float alpha = 1.0;
    const float beta = 0.0;
    cublasGemmAlgo_t CuBlasALG = static_cast<cublasGemmAlgo_t>(0);
    // Tensor core enabled
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    cudaDeviceSynchronize();
    for (int i = 0; i < WARM_UP_ITERATION; i++) {
        cublas_status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, A, CUDA_R_16F, k, B, CUDA_R_16F, k, &beta, D_cublas,
                                     CUDA_R_16F, m, CUDA_R_32F, CuBlasALG);
        checkCublasError(cublas_status, __LINE__);
    }
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATION; i++)
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, A, CUDA_R_16F, k, B, CUDA_R_16F, k, &beta, D_cublas, CUDA_R_16F, m,
                     CUDA_R_32F, CuBlasALG);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    //
    float milliseconds_cublas_tc = 0;
    cudaEventElapsedTime(&milliseconds_cublas_tc, start, stop);
    milliseconds_cublas_tc = milliseconds_cublas_tc / BENCHMARK_ITERATION;
    float tflops_cublas_tc = static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_cublas_tc / 1000.)) / 1e12;
    half *D_cublas_h = NULL; // col major
    D_cublas_h = (half *)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (D_cublas_h == NULL) {
        printf("Error in spmm_test.cu: line %d CPU Malloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemcpy(D_cublas_h, D_cublas, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost); // Col Major
    cudaFree(D_cublas);
    /////////////////////////////////////////////////////////////////////////////////////////////////

    auto Split_K = SPLIT_K;

    // SpInfer
    ////////////////////////////////////////////////////////////////////////////////////////////////
    half *D_SpMM_bitmapv3 = NULL;
    cudaMalloc(reinterpret_cast<void **>(&D_SpMM_bitmapv3), sizeof(half) * M_GLOBAL * N_GLOBAL);
    if (D_SpMM_bitmapv3 == NULL) {
        printf("Error in spmm_test.cu: line %d cudaMalloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemset(D_SpMM_bitmapv3, 0, sizeof(half) * M_GLOBAL * N_GLOBAL);

    // Define the output pointer
    half *Compressed_Val_cpu_v3 = nullptr;
    int *bitmap_TileOffsets_cpu_v3 = nullptr;
    int *bitmap_TileOffsets_median_cpu_v3 = nullptr;
    int *bitmap_TileOffsets_global_cpu_v3 = nullptr;
    uint64_t *bitmap_cpu_v3 = nullptr;
    int max_nnz_intilev3 = 0;
    // Call the InitSparseMatrixA_bitmap_v6 function
    auto num_gtilesv3 =
        InitSparseMatrixA_bitmap_v6(A_h, M_GLOBAL, K_GLOBAL, 8, 16, 64, 8, 64, 64, &Compressed_Val_cpu_v3, &bitmap_TileOffsets_cpu_v3,
                                    &bitmap_TileOffsets_median_cpu_v3, &bitmap_TileOffsets_global_cpu_v3, &bitmap_cpu_v3, max_nnz_intilev3);

//     print_bitmap_v3_results(Compressed_Val_cpu_v3, bitmap_TileOffsets_cpu_v3, bitmap_TileOffsets_global_cpu_v3, bitmap_cpu_v3, num_gtilesv3,
//                             num_gtilesv3, max_nnz_intilev3);

    auto local_tile_numv3 = 8 * 8;
    auto median_tile_numv3 = 4 * 1;
    auto num_ltilesv3 = num_gtilesv3 * local_tile_numv3;
    auto num_mtilesv3 = num_gtilesv3 * median_tile_numv3;
    // The offset of the last tile is equal to the total number of compressed non-zero values
    int val_count_v3 = bitmap_TileOffsets_global_cpu_v3[num_gtilesv3];
    int val_count_median_v3 = bitmap_TileOffsets_median_cpu_v3[num_mtilesv3];
    // Adjust max_nnz_intilev3 to a multiple of 64
    if (max_nnz_intilev3 % 64 != 0) {
        max_nnz_intilev3 = ((max_nnz_intilev3 / 64) + 1) * 64;
    }
    printf("num_global_tiles: %d, bitmap v3 NNZ: %d, bitmap v3 median layer NNZ: %d,  max_nnz_intilev3: %d \n", num_gtilesv3, val_count_v3,
           val_count_median_v3, max_nnz_intilev3);
    half *Compressed_Val_gpu_v3 = nullptr;
    int *bitmap_TileOffsets_gpu_v3 = nullptr;
    int *bitmap_TileOffsets_median_gpu_v3 = nullptr;
    int *bitmap_TileOffsets_global_gpu_v3 = nullptr;
    uint64_t *bitmap_gpu_v3 = nullptr;
    cudaMalloc(&bitmap_TileOffsets_gpu_v3, sizeof(int) * (num_ltilesv3 + 1)); // for (16*64 tile specific)
    cudaMalloc(&bitmap_gpu_v3, sizeof(uint64_t) * (num_ltilesv3));
    cudaMalloc(&bitmap_TileOffsets_median_gpu_v3, sizeof(int) * (num_mtilesv3));
    cudaMalloc(&bitmap_TileOffsets_global_gpu_v3, sizeof(int) * (num_gtilesv3 + 1));
    if (val_count_v3 == 0)
        val_count_v3 = 1; // For 100% sparsity, NNZ = 0, malloc will return NULL
    cudaMalloc(&Compressed_Val_gpu_v3, sizeof(half) * val_count_v3);
    if (bitmap_TileOffsets_gpu_v3 == NULL || bitmap_gpu_v3 == NULL || Compressed_Val_gpu_v3 == NULL || bitmap_TileOffsets_global_gpu_v3 == NULL) {
        printf("Error in malloc memory from device memory!\n");
        exit(-1);
    }
    cudaMemcpy(bitmap_TileOffsets_gpu_v3, bitmap_TileOffsets_cpu_v3, sizeof(int) * (num_ltilesv3 + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(bitmap_TileOffsets_global_gpu_v3, bitmap_TileOffsets_global_cpu_v3, sizeof(int) * (num_gtilesv3 + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(bitmap_TileOffsets_median_gpu_v3, bitmap_TileOffsets_median_cpu_v3, sizeof(int) * (num_mtilesv3), cudaMemcpyHostToDevice);
    cudaMemcpy(bitmap_gpu_v3, bitmap_cpu_v3, sizeof(uint64_t) * num_ltilesv3, cudaMemcpyHostToDevice);
    cudaMemcpy(Compressed_Val_gpu_v3, Compressed_Val_cpu_v3, sizeof(half) * val_count_v3, cudaMemcpyHostToDevice);
    free(bitmap_TileOffsets_cpu_v3);
    free(bitmap_cpu_v3);
    free(Compressed_Val_cpu_v3);
    free(bitmap_TileOffsets_global_cpu_v3);
    free(bitmap_TileOffsets_median_cpu_v3);
    printf("Done! Compressed A matrix for bitmap v3 GPU kernel.\n");

    // Compress B matrix similar to A
    printf("Compressing B matrix...\n");
    half *Compressed_B_cpu_v3 = nullptr;
    int *B_bitmap_TileOffsets_cpu_v3 = nullptr;
    int *B_bitmap_TileOffsets_median_cpu_v3 = nullptr;
    int *B_bitmap_TileOffsets_global_cpu_v3 = nullptr;
    uint64_t *B_bitmap_cpu_v3 = nullptr;
    int B_max_nnz_intilev3 = 0;

    // Call the InitSparseMatrixA_bitmap_v6 function for B (notice B is K_GLOBAL x N_GLOBAL, in column-major order)
    auto B_num_gtilesv3 =
        InitSparseMatrixA_bitmap_v6_B(B_h, K_GLOBAL, N_GLOBAL, 8, 16, 64, 8, 16, 64, &Compressed_B_cpu_v3, &B_bitmap_TileOffsets_cpu_v3,
                                      &B_bitmap_TileOffsets_median_cpu_v3, &B_bitmap_TileOffsets_global_cpu_v3, &B_bitmap_cpu_v3, B_max_nnz_intilev3);
    auto B_local_tile_numv3 = 8 * 8;
    auto B_median_tile_numv3 = 4 * 1;
    auto B_num_ltilesv3 = B_num_gtilesv3 * B_local_tile_numv3;
    auto B_num_mtilesv3 = B_num_gtilesv3 * B_median_tile_numv3;

    // The offset of the last tile is equal to the total number of compressed non-zero values
    int B_val_count_v3 = B_bitmap_TileOffsets_global_cpu_v3[B_num_gtilesv3];
    int B_val_count_median_v3 = B_bitmap_TileOffsets_median_cpu_v3[B_num_mtilesv3];

    // Adjust B_max_nnz_intilev3 to a multiple of 64
    if (B_max_nnz_intilev3 % 64 != 0) {
        B_max_nnz_intilev3 = ((B_max_nnz_intilev3 / 64) + 1) * 64;
    }

    printf("B num_global_tiles: %d, bitmap v3 NNZ: %d, bitmap v3 median layer NNZ: %d, max_nnz_intilev3: %d \n", B_num_gtilesv3, B_val_count_v3,
           B_val_count_median_v3, B_max_nnz_intilev3);

    // Allocate device memory for compressed B
    half *Compressed_B_gpu_v3 = nullptr;
    int *B_bitmap_TileOffsets_gpu_v3 = nullptr;
    int *B_bitmap_TileOffsets_median_gpu_v3 = nullptr;
    int *B_bitmap_TileOffsets_global_gpu_v3 = nullptr;
    uint64_t *B_bitmap_gpu_v3 = nullptr;

    cudaMalloc(&B_bitmap_TileOffsets_gpu_v3, sizeof(int) * (B_num_ltilesv3 + 1));
    cudaMalloc(&B_bitmap_gpu_v3, sizeof(uint64_t) * (B_num_ltilesv3));
    cudaMalloc(&B_bitmap_TileOffsets_median_gpu_v3, sizeof(int) * (B_num_mtilesv3));
    cudaMalloc(&B_bitmap_TileOffsets_global_gpu_v3, sizeof(int) * (B_num_gtilesv3 + 1));

    if (B_val_count_v3 == 0)
        B_val_count_v3 = 1; // For 100% sparsity, NNZ = 0, malloc will return NULL

    cudaMalloc(&Compressed_B_gpu_v3, sizeof(half) * B_val_count_v3);

    if (B_bitmap_TileOffsets_gpu_v3 == NULL || B_bitmap_gpu_v3 == NULL || Compressed_B_gpu_v3 == NULL || B_bitmap_TileOffsets_global_gpu_v3 == NULL) {
        printf("Error in malloc memory from device memory for compressed B!\n");
        exit(-1);
    }

    // Copy compressed B data to device
    cudaMemcpy(B_bitmap_TileOffsets_gpu_v3, B_bitmap_TileOffsets_cpu_v3, sizeof(int) * (B_num_ltilesv3 + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(B_bitmap_TileOffsets_global_gpu_v3, B_bitmap_TileOffsets_global_cpu_v3, sizeof(int) * (B_num_gtilesv3 + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(B_bitmap_TileOffsets_median_gpu_v3, B_bitmap_TileOffsets_median_cpu_v3, sizeof(int) * (B_num_mtilesv3), cudaMemcpyHostToDevice);
    cudaMemcpy(B_bitmap_gpu_v3, B_bitmap_cpu_v3, sizeof(uint64_t) * B_num_ltilesv3, cudaMemcpyHostToDevice);
    cudaMemcpy(Compressed_B_gpu_v3, Compressed_B_cpu_v3, sizeof(half) * B_val_count_v3, cudaMemcpyHostToDevice);

    // Free CPU memory for compressed B
    free(B_bitmap_TileOffsets_cpu_v3);
    free(B_bitmap_cpu_v3);
    free(Compressed_B_cpu_v3);
    free(B_bitmap_TileOffsets_global_cpu_v3);
    free(B_bitmap_TileOffsets_median_cpu_v3);

    printf("Done! Compressed B matrix for bitmap v3 GPU kernel.\n");

    printf("Launching bitmapv3 without Ahead of Time Sparse Data Reordering...\n");
    Split_K = SPLIT_K;
    printf("Split_K = %d\n", Split_K);
    half *Reduction_Workspace_bitmapv3 = NULL;
    cudaMalloc(reinterpret_cast<void **>(&Reduction_Workspace_bitmapv3), sizeof(half) * M_GLOBAL * N_GLOBAL * Split_K);
    if (Reduction_Workspace_bitmapv3 == NULL) {
        printf("Error in cudaMalloc\n");
        exit(-1);
    }
    int *max_nnz_intilev3_gpu = nullptr;
    cudaMalloc(&max_nnz_intilev3_gpu, sizeof(int));
    if (max_nnz_intilev3_gpu == NULL) {
        printf("Error in cudaMalloc for max_nnz_intilev3_gpu\n");
        exit(-1);
    }
    cudaMemcpy(max_nnz_intilev3_gpu, &max_nnz_intilev3, sizeof(int), cudaMemcpyHostToDevice);

    int *B_max_nnz_intilev3_gpu = nullptr;
    cudaMalloc(&B_max_nnz_intilev3_gpu, sizeof(int));
    if (B_max_nnz_intilev3_gpu == NULL) {
        printf("Error in cudaMalloc for B_max_nnz_intilev3_gpu\n");
        exit(-1);
    }
    cudaMemcpy(B_max_nnz_intilev3_gpu, &B_max_nnz_intilev3, sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < WARM_UP_ITERATION; i++)
        SpMM_SplitK_API_bitmap_v3(0, A,
                                  Compressed_Val_gpu_v3,            // half
                                  bitmap_TileOffsets_global_gpu_v3, // int
                                  bitmap_TileOffsets_median_gpu_v3, // int
                                  bitmap_gpu_v3,                    // uint64
                                  max_nnz_intilev3_gpu,             // int
                                  B, Compressed_B_gpu_v3, B_bitmap_TileOffsets_global_gpu_v3, B_bitmap_TileOffsets_median_gpu_v3, B_bitmap_gpu_v3,
                                  B_max_nnz_intilev3_gpu, D_SpMM_bitmapv3, M_GLOBAL, N_GLOBAL, K_GLOBAL, Reduction_Workspace_bitmapv3, Split_K);
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATION; i++)
        SpMM_SplitK_API_bitmap_v3(0, A, Compressed_Val_gpu_v3, bitmap_TileOffsets_global_gpu_v3, bitmap_TileOffsets_median_gpu_v3, bitmap_gpu_v3,
                                  max_nnz_intilev3_gpu, B, Compressed_B_gpu_v3, B_bitmap_TileOffsets_global_gpu_v3,
                                  B_bitmap_TileOffsets_median_gpu_v3, B_bitmap_gpu_v3, B_max_nnz_intilev3_gpu, D_SpMM_bitmapv3, M_GLOBAL, N_GLOBAL,
                                  K_GLOBAL, Reduction_Workspace_bitmapv3, Split_K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    checkLastCudaError(__LINE__);
    // //
    float milliseconds_SpMM_bitmapv3 = 0.0f;
    cudaEventElapsedTime(&milliseconds_SpMM_bitmapv3, start, stop);
    milliseconds_SpMM_bitmapv3 = milliseconds_SpMM_bitmapv3 / BENCHMARK_ITERATION;
    float tflops_SpMM_bitmapv3 =
        static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_SpMM_bitmapv3 / 1000.)) / 1e12;
    half *D_SpMM_hbitmapv3 = NULL; // col major
    D_SpMM_hbitmapv3 = (half *)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    cudaMemcpy(D_SpMM_hbitmapv3, D_SpMM_bitmapv3, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost); // Col Major
    cudaFree(D_SpMM_bitmapv3);
    cudaFree(bitmap_TileOffsets_gpu_v3);
    cudaFree(bitmap_TileOffsets_global_gpu_v3);
    cudaFree(bitmap_TileOffsets_median_gpu_v3);
    cudaFree(bitmap_gpu_v3);
    cudaFree(Compressed_Val_gpu_v3);
    cudaFree(Reduction_Workspace_bitmapv3);
    cudaFree(max_nnz_intilev3_gpu);

    double totalError_SpMM_bitmapv3 = 0.0;

    totalError_SpMM_bitmapv3 = ComputeTotalError(D_cublas_h, D_SpMM_hbitmapv3, M_GLOBAL, N_GLOBAL);

    free(D_SpMM_hbitmapv3);

    PrintPerformance("SpInfer", milliseconds_SpMM_bitmapv3, tflops_SpMM_bitmapv3, totalError_SpMM_bitmapv3);
    PrintPerformance("CuBlas_TC", milliseconds_cublas_tc, tflops_cublas_tc, 0.0);

    free(D_cublas_h);
    free(A_h);
    free(B_h);

    cudaFree(A);
    cudaFree(B);

    return 0;
}
