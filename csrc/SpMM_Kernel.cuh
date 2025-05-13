/***************************************************************************
 * Copyright 2025 The SpInfer Authors. All rights reserved.
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
#include "MatMulUtilities.cuh"
#include <inttypes.h>
#include <vector>
#define __STDC_FORMAT_MACROS

template <typename TilingConfig>
__global__ void SpMM_Kernel_bitmap_v3(const half *A, const half *Compressed_A,
                                      const int *TileOffsets, const int *TileOffsets_Median,
                                      const uint64_t *bitmap, const int *ptr_max_nnz_intile,
                                      const half *B, const half *Compressed_B,
                                      const int *TileOffsets_B, const int *TileOffsets_Median_B,
                                      const uint64_t *bitmap_B, const int *max_nnz_intile_B,
                                      half *Reduction_Workspace, const int M_Global,
                                      const int N_Global, const int K_Global, int Split_K)
{
    int max_nnz_intile = *ptr_max_nnz_intile;
    int max_nnz_intile_B_ = *max_nnz_intile_B;
    const int BatchID = blockIdx.y / (M_Global / TilingConfig::TILE_M);
    const int IsLastBatch = (BatchID == (Split_K - 1));
    const int x = blockIdx.x;
    const int y = blockIdx.y % (M_Global / TilingConfig::TILE_M);
    //
    const int NumKBlock = K_Global / TILE_K; // assert (K_Global%TILE_K==0);
    const int AverageNumKBlock = (NumKBlock - 1) / Split_K + 1;
    const int RoundedKBlock = AverageNumKBlock * Split_K;
    const int PaddingKBlock = RoundedKBlock - NumKBlock;
    int NumIter = 0;
    if (IsLastBatch)
        NumIter = AverageNumKBlock - PaddingKBlock;
    else
        NumIter = AverageNumKBlock;
    const int *TileOffsets_ThisBlock = nullptr;
    const int BlockOffset = K_Global / TILE_K * y + BatchID * AverageNumKBlock;
    TileOffsets_ThisBlock = TileOffsets + BlockOffset;

    const int *TileOffsets_ThisBlock_B = nullptr;
    const int BlockOffset_B = K_Global / TILE_K * x + BatchID * AverageNumKBlock;
    TileOffsets_ThisBlock_B = TileOffsets_B + BlockOffset_B;

    ////////
    extern __shared__ __align__(128) half smem[]; // at least be 128 Bytes aligned
    half *smem_B = &smem[max_nnz_intile];
    uint64_t *smem_Bitmap =
        reinterpret_cast<uint64_t *>(&smem[max_nnz_intile + max_nnz_intile_B_ * 2]);
    uint64_t *smem_Bitmap_B =
        reinterpret_cast<uint64_t *>(&smem[max_nnz_intile + max_nnz_intile_B_ * 2 + 512]);

    // Warp and lane identification.
    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const int Tile_Start_M = y * TilingConfig::TILE_M;
    const int Tile_Start_Bitmap = y * TilingConfig::TILE_BITMAP_M_V3;
    const int Tile_Start_Bitmap_B = x * TilingConfig::TILE_BITMAP_M_V3;
    const int Tile_Start_N = x * TilingConfig::TILE_N;
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Compute a grid of C matrix tiles in each warp.
    int Warp_i = warpId / TilingConfig::BLOCK_COL_WARPS;
    int Warp_j = warpId % TilingConfig::BLOCK_COL_WARPS;
    int warp_start_row = WARP_ROW_TENSORS_BITMAP_V3 * MMA_M * Warp_i;
    int warp_start_col = TilingConfig::WARP_COL_TENSORS * MMA_N * Warp_j;
    uint32_t __restrict__ a[WARP_ROW_TENSORS_BITMAP_V3 * BLOCK_K_TENSORS][4];
    //     uint32_t __restrict__ b[TilingConfig::WARP_COL_TENSORS * 2][4];
    uint32_t __restrict__ b_[TilingConfig::WARP_COL_TENSORS * 2][4];
    uint64_t *smem_BitmapWarp = smem_Bitmap + Warp_i * 16;
    //     uint64_t *smem_BitmapWarp_B = smem_Bitmap + Warp_i * 16;
    const int *TileOffsets_ThisWarp = nullptr;
    const int WarpOffset = BlockOffset * 4 + Warp_i;
    TileOffsets_ThisWarp = TileOffsets_Median + WarpOffset;

    // gld addr of copying B tile from GlobalMemory to SharedMemory
    const uint64_t *BitmapTileGlobalPTR = bitmap + Tile_Start_Bitmap * K_Global + BatchID * AverageNumKBlock * TilingConfig::TILE_BITMAP_K_V3;
    const uint64_t *BitmapTileGlobalPTR_B = bitmap_B + Tile_Start_Bitmap_B * K_Global + BatchID * AverageNumKBlock * TilingConfig::TILE_BITMAP_K_V3;

    // copy A val and bitmap to shared memory
    CopyTileFromGlobalToShared_Bitmap_1_64<TilingConfig::TILE_BITMAP_M_V3, TilingConfig>(
        smem_Bitmap, BitmapTileGlobalPTR);
    CopyTileFromGlobalToShared_Sparse<TilingConfig>(smem, Compressed_A + TileOffsets_ThisBlock[0],
                                                    TileOffsets_ThisBlock[0 + 1] -
                                                        TileOffsets_ThisBlock[0]);
    cp_async_group_commit();

    // copy B val and bitmap to shared memory
    CopyTileFromGlobalToShared_Bitmap_1_64<TilingConfig::TILE_BITMAP_M_V3, TilingConfig>(
        smem_Bitmap_B, BitmapTileGlobalPTR_B);
    CopyTileFromGlobalToShared_Sparse<TilingConfig>(
        smem_B, Compressed_B + TileOffsets_ThisBlock_B[0],
        TileOffsets_ThisBlock_B[0 + 1] - TileOffsets_ThisBlock_B[0]);
    cp_async_group_commit();

    // Initilazing C Matrix to Zeros.
    float c[WARP_ROW_TENSORS_BITMAP_V3 * TilingConfig::WARP_COL_TENSORS][REG_PER_C_TENSOR_16_16];
    for (int i = 0; i < WARP_ROW_TENSORS_BITMAP_V3 * TilingConfig::WARP_COL_TENSORS; i++)
        for (int j = 0; j < REG_PER_C_TENSOR_16_16; j++)
            c[i][j] = 0.0f;

    cp_async_wait_group<1>(); // bitmap loading done
    __syncthreads();

    SpMM_LoadFragAwithBitmapFromShem(a, smem + TileOffsets_ThisWarp[0], smem_BitmapWarp);

    cp_async_wait_group<0>(); // bitmap loading done
    __syncthreads();

    int current_sparse_tile_start = TileOffsets_ThisBlock[1];
    int current_sparse_tile_nnz = TileOffsets_ThisBlock[1 + 1] - TileOffsets_ThisBlock[1];

    int current_sparse_tile_start_B = TileOffsets_ThisBlock_B[1];
    int current_sparse_tile_nnz_B = TileOffsets_ThisBlock_B[1 + 1] - TileOffsets_ThisBlock_B[1];

// Go through the global K dimension by a fixed step at a time.
// write buffer[1] first, read buffer[0] first
#pragma unroll(1)
    for (int tile_id_k = 0; tile_id_k < NumIter; tile_id_k++)
    {
        BitmapTileGlobalPTR = BitmapTileGlobalPTR + TilingConfig::TILE_BITMAP_K_V3;
        BitmapTileGlobalPTR_B = BitmapTileGlobalPTR_B + TilingConfig::TILE_BITMAP_K_V3;

        current_sparse_tile_start = TileOffsets_ThisBlock[tile_id_k + 1];
        current_sparse_tile_nnz =
            TileOffsets_ThisBlock[tile_id_k + 1 + 1] - TileOffsets_ThisBlock[tile_id_k + 1];

        current_sparse_tile_start_B = TileOffsets_ThisBlock_B[tile_id_k + 1];
        current_sparse_tile_nnz_B =
            TileOffsets_ThisBlock_B[tile_id_k + 1 + 1] - TileOffsets_ThisBlock_B[tile_id_k + 1];

        // double buffer
        half *__restrict__ smem_write_B_PTR = smem_B;
        half *__restrict__ smem_read_B_PTR = smem_B;
        smem_write_B_PTR =
            smem_B + ((tile_id_k + 1) % 2) * (max_nnz_intile_B_); // The current write address of B
        smem_read_B_PTR =
            smem_B + ((tile_id_k) % 2) * (max_nnz_intile_B_); // The current reading address of B

        uint64_t *__restrict__ smem_Bitmap_B_write_B_PTR = smem_Bitmap_B;
        uint64_t *__restrict__ smem_Bitmap_B_read_B_PTR = smem_Bitmap_B;
        smem_Bitmap_B_write_B_PTR =
            smem_Bitmap_B + ((tile_id_k + 1) % 2) * (64); // The current write address of B
        smem_Bitmap_B_read_B_PTR =
            smem_Bitmap_B + ((tile_id_k) % 2) * (64); // The current reading address of B

        bool GlobalCopy = (tile_id_k + 1) < NumIter;

        // copy A val and bitmap to shared memory
        CopyTileFromGlobalToShared_Bitmap_1_64<TilingConfig::TILE_BITMAP_M_V3, TilingConfig>(
            smem_Bitmap, BitmapTileGlobalPTR, GlobalCopy);
        CopyTileFromGlobalToShared_Sparse<TilingConfig>(
            smem, Compressed_A + current_sparse_tile_start, current_sparse_tile_nnz, GlobalCopy);
        cp_async_group_commit();

        // copy B val and bitmap to shared memory
        CopyTileFromGlobalToShared_Bitmap_1_64<TilingConfig::TILE_BITMAP_M_V3, TilingConfig>(
            smem_Bitmap_B_write_B_PTR, BitmapTileGlobalPTR_B, GlobalCopy);
        CopyTileFromGlobalToShared_Sparse<TilingConfig>(smem_write_B_PTR,
                                                        Compressed_B + current_sparse_tile_start_B,
                                                        current_sparse_tile_nnz_B, GlobalCopy);
        cp_async_group_commit();

        PipelinedCoreComputationsBitmap<TilingConfig>(c, a, b_, smem_read_B_PTR, warp_start_row,
                                                      warp_start_col, smem_Bitmap_B_read_B_PTR);

        cp_async_wait_group<1>();
        __syncthreads();

        SpMM_LoadFragAwithBitmapFromShem(a, smem + TileOffsets_ThisWarp[(tile_id_k + 1) * 4],
                                         smem_BitmapWarp, GlobalCopy);
        cp_async_wait_group<0>(); // Sync to ensure the completion of Loading B to shared memory
        __syncthreads();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Store the C fragments to shared memory.
    float (*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C] =
        reinterpret_cast<float (*)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C]>(smem);
    StoreToSharedMemoryFromRegisterBitmapV3<TilingConfig>(smem_CFrag, c);

    // Now that shared memory contains all the D tiles, stream them to global memory.
    half *BlockGlobalPTR = Reduction_Workspace + BatchID * (M_Global * N_Global) + Tile_Start_M + Tile_Start_N * M_Global;
#pragma unroll
    for (int i = warpId; i < TilingConfig::TILE_N2; i += TilingConfig::BLOCK_WARPS)
#pragma unroll
        for (int j = threadIdx.x % WARP_SIZE; j < TilingConfig::TILE_M; j += WARP_SIZE) // j-th row
            BlockGlobalPTR[j + i * M_Global] = __float2half_rn((*(smem_CFrag + i))[j]);
}
