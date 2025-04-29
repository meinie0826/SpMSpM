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
__global__ void SpMM_Kernel_bitmap_v3(const half *A, const half *Compressed_A, const int *TileOffsets, const int *TileOffsets_Median,
                                      const uint64_t *bitmap, const int *ptr_max_nnz_intile, const half *B, const half *Compressed_B,
                                      const int *TileOffsets_B, const int *TileOffsets_Median_B, const uint64_t *bitmap_B,
                                      const int *max_nnz_intile_B, half *Reduction_Workspace, const int M_Global, const int N_Global,
                                      const int K_Global, int Split_K) {
    int max_nnz_intile = *ptr_max_nnz_intile;
    int max_nnz_intile_B_ = *max_nnz_intile_B;

    const int x = blockIdx.x;
    const int y = blockIdx.y % (M_Global / TilingConfig::TILE_M);
    //
    const int NumKBlock = K_Global / TILE_K; // assert (K_Global%TILE_K==0);
    const int AverageNumKBlock = NumKBlock;

    int NumIter = 0;

    NumIter = AverageNumKBlock;
    const int *TileOffsets_ThisBlock = nullptr;
    const int BlockOffset = K_Global / TILE_K * y;
    TileOffsets_ThisBlock = TileOffsets + BlockOffset;

    const int *TileOffsets_ThisBlock_B = nullptr;
    const int BlockOffset_B = K_Global / TILE_K * x;
    TileOffsets_ThisBlock_B = TileOffsets_B + BlockOffset_B;

    ////////
    extern __shared__ __align__(128) half smem[]; // at least be 128 Bytes aligned
    uint64_t *smem_Bitmap = reinterpret_cast<uint64_t *>(&smem[max_nnz_intile + (TILE_K * TilingConfig::TILE_N)]);
    half *smem_B = &smem[max_nnz_intile];
    half *smem_B_ = &smem[max_nnz_intile + (TILE_K * TilingConfig::TILE_N) + 512];
    uint64_t *smem_Bitmap_B = reinterpret_cast<uint64_t *>(&smem_B_[max_nnz_intile_B_]);
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
    uint32_t __restrict__ b[TilingConfig::WARP_COL_TENSORS * 2][4];
    uint32_t __restrict__ b_[TilingConfig::WARP_COL_TENSORS * 2][4];
    uint64_t *smem_BitmapWarp = smem_Bitmap + Warp_i * 16;
    //     uint64_t *smem_BitmapWarp_B = smem_Bitmap + Warp_i * 16;
    const int *TileOffsets_ThisWarp = nullptr;
    const int WarpOffset = BlockOffset * 4 + Warp_i;
    TileOffsets_ThisWarp = TileOffsets_Median + WarpOffset;

    const int *TileOffsets_ThisWarp_B = TileOffsets_Median_B + WarpOffset;
    // gld addr of copying B tile from GlobalMemory to SharedMemory
    const half *BTileGlobalPTR = B + Tile_Start_N * K_Global;
    const uint64_t *BitmapTileGlobalPTR = bitmap + Tile_Start_Bitmap * K_Global;
    const uint64_t *BitmapTileGlobalPTR_B = bitmap_B + Tile_Start_Bitmap_B * K_Global;

    // Initilazing C Matrix to Zeros.
    float c[WARP_ROW_TENSORS_BITMAP_V3 * TilingConfig::WARP_COL_TENSORS][REG_PER_C_TENSOR_16_16];
    for (int i = 0; i < WARP_ROW_TENSORS_BITMAP_V3 * TilingConfig::WARP_COL_TENSORS; i++)
        for (int j = 0; j < REG_PER_C_TENSOR_16_16; j++)
            c[i][j] = 0.0f;

    int current_sparse_tile_start = TileOffsets_ThisBlock[0];
    int current_sparse_tile_nnz = TileOffsets_ThisBlock[0 + 1] - TileOffsets_ThisBlock[0];

    int current_sparse_tile_start_B = TileOffsets_ThisBlock_B[0];
    int current_sparse_tile_nnz_B = TileOffsets_ThisBlock_B[0 + 1] - TileOffsets_ThisBlock_B[0];

    // Go through the global K dimension by a fixed step at a time.
    // write buffer[1] first, read buffer[0] first

    for (int tile_id_k = 0; tile_id_k < NumIter; tile_id_k++) {
        // Copying next Bitmap Tile to write shem
        CopyTileFromGlobalToShared_Bitmap_1_64<TilingConfig::TILE_BITMAP_M_V3, TilingConfig>(
            smem_Bitmap, BitmapTileGlobalPTR, true); // Load the 2*8 bitmap after the double buffer B shared tile

        // Copying next B Tile to write shem
        CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(smem_B, BTileGlobalPTR, K_Global, true);

        CopyTileFromGlobalToShared_Bitmap_1_64<TilingConfig::TILE_BITMAP_M_V3, TilingConfig>(
            smem_Bitmap_B, BitmapTileGlobalPTR_B, true); // Load the 2*8 bitmap after the double buffer B shared tile

        // Copying next Sparse A Tile to write shem
        CopyTileFromGlobalToShared_Sparse<TilingConfig>(smem, Compressed_A + current_sparse_tile_start, current_sparse_tile_nnz, true);
        CopyTileFromGlobalToShared_Sparse<TilingConfig>(smem_B_, Compressed_B + current_sparse_tile_start_B, current_sparse_tile_nnz_B, true);
        // __syncthreads();
        // // // 添加调试打印
        // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        //     printf("\n=== smem_B (64x64) ===\n");
        //     for (int i = 0; i < 64; i++) {
        //         for (int j = 0; j < 64; j++) {
        //             printf("%.2f ", __half2float(smem_B[i * 64 + j]));
        //             if (j % 16 == 15)
        //                 printf("\n");
        //         }
        //         printf("\n");
        //     }

        //     printf("\n=== smem_+++_ (64x64) ===\n");
        //     for (int i = 0; i < 64; i++) {
        //         for (int j = 0; j < 64; j++) {
        //             printf("%.2f ", __half2float(smem_B_[i * 64 + j]));
        //             if (j % 16 == 15)
        //                 printf("\n");
        //         }
        //         printf("\n");
        //     }

        //     printf("\n=== smem_A (64x64) ===\n");
        //     for (int i = 0; i < 64; i++) {
        //         for (int j = 0; j < 64; j++) {
        //             printf("%.2f ", __half2float(smem[i * 64 + j]));
        //             if (j % 16 == 15)
        //                 printf("\n");
        //         }
        //         printf("\n");
        //     }
        // }
        // __syncthreads();


        SpMM_LoadFragAwithBitmapFromShem(a, smem + TileOffsets_ThisWarp[(tile_id_k) * 4], smem_BitmapWarp, true);
        //SpMM_LoadFragAwithBitmapFromShem_B(b_, smem_B_, smem_Bitmap_B, TileOffsets_ThisWarp, 0, true);

        PipelinedCoreComputationsBitmap<TilingConfig>(c, a, b, smem_B, warp_start_row, warp_start_col, smem_Bitmap_B, TileOffsets_ThisWarp_B,
                                                      tile_id_k, b_, smem_B_);
        BitmapTileGlobalPTR = BitmapTileGlobalPTR + TilingConfig::TILE_BITMAP_K_V3;
        BitmapTileGlobalPTR_B = BitmapTileGlobalPTR_B + TilingConfig::TILE_BITMAP_K_V3;
        BTileGlobalPTR = BTileGlobalPTR + TILE_K;
        current_sparse_tile_start = TileOffsets_ThisBlock[tile_id_k + 1];
        current_sparse_tile_nnz = TileOffsets_ThisBlock[tile_id_k + 1 + 1] - TileOffsets_ThisBlock[tile_id_k + 1];

        current_sparse_tile_start_B = TileOffsets_ThisBlock_B[tile_id_k + 1];
        current_sparse_tile_nnz_B = TileOffsets_ThisBlock_B[tile_id_k + 1 + 1] - TileOffsets_ThisBlock_B[tile_id_k + 1];
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Store the C fragments to shared memory.
    float (*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C] =
        reinterpret_cast<float (*)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C]>(smem);
    StoreToSharedMemoryFromRegisterBitmapV3<TilingConfig>(smem_CFrag, c);

    // Now that shared memory contains all the D tiles, stream them to global memory.
    half *BlockGlobalPTR = Reduction_Workspace + Tile_Start_M + Tile_Start_N * M_Global;

    for (int i = warpId; i < TilingConfig::TILE_N2; i += TilingConfig::BLOCK_WARPS)     // i-th column
        for (int j = threadIdx.x % WARP_SIZE; j < TilingConfig::TILE_M; j += WARP_SIZE) // j-th row
            BlockGlobalPTR[j + i * M_Global] = __float2half_rn((*(smem_CFrag + i))[j]);
}
