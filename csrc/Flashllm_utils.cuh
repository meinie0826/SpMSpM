#ifndef THIRD_PARTY_FLASHLLM_UTILS_H_
#define THIRD_PARTY_FLASHLLM_UTILS_H_
#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace flash_llm_utils {
    

// Fixed Parameters
#define MMA_M 16
#define MMA_N 16
#define MMA_K 16
#define WARP_SIZE 32
// Unchangable
#define WARP_ROW_TENSORS 4
#define BLOCK_K_TENSORS 4
#define TILE_K (MMA_K * BLOCK_K_TENSORS)  // 64
// Parameters for copying A_TILE & B_TILE & C_TILE
#define COPY_UNIT_FP16_ROWS 8
#define COPY_UNIT_FP16_COLS 64
#define HALF_PER_128B 8           // LDS.128 -> 8 * FP16
#define REG_PER_C_TENSOR_16_16 8  // 8 for FP32 Accumulation; 4 for FP16 Accumulation

#define PADDING_SHARED_MEM_FOR_C 4  // Padding 8/2 float each column to eliminating bank-conflict in C fragments

template<int BLOCK_ROW_WARPS_, int BLOCK_COL_WARPS_, int WARP_COL_TENSORS_, int N8_ = 0>
struct TilingConfig {
    static constexpr int BLOCK_ROW_WARPS  = BLOCK_ROW_WARPS_;
    static constexpr int BLOCK_COL_WARPS  = BLOCK_COL_WARPS_;
    static constexpr int WARP_COL_TENSORS = WARP_COL_TENSORS_;
    // Sanity checks on the template arguments.
    // static_assert((BLOCK_ROW_WARPS * BLOCK_COL_WARPS) == 4,
    //               "The number of WARPS per threadblock must be 4.");
    // Derived Parameters
    static constexpr int TILE_M        = MMA_M * (WARP_ROW_TENSORS * BLOCK_ROW_WARPS);
    static constexpr int TILE_N        = MMA_N * (WARP_COL_TENSORS * BLOCK_COL_WARPS);
    static constexpr int BLOCK_WARPS   = BLOCK_ROW_WARPS * BLOCK_COL_WARPS;
    static constexpr int BLOCK_THREADS = BLOCK_WARPS * WARP_SIZE;
    // temporary implementation to support N=8
    static constexpr int N8      = N8_;
    static constexpr int TILE_N2 = N8 ? 8 : TILE_N;
};

template<int NUM_REG_FOR_SPARSE_KERNEL_ = 64>
struct SparseKernelConfig {
    static constexpr int NUM_REG_FOR_SPARSE_KERNEL    = NUM_REG_FOR_SPARSE_KERNEL_;
    static constexpr int VECTOR_SIZE                  = 4;
    static constexpr int PADDING_SIZE_FOR_TILEOFFSETS = 2;  // (N+1 offsets) + 1 padding
    // Sanity checks on the template arguments.
    // static_assert((BLOCK_ROW_WARPS * BLOCK_COL_WARPS) == 4,
    //               "The number of WARPS per threadblock must be 4.");
    // Derived Parameters
    // static constexpr int TILE_M = MMA_M * (WARP_ROW_TENSORS * BLOCK_ROW_WARPS);
};


template<int SizeInBytes>
__device__ __forceinline__ void cp_async(half* smem_ptr, const half* global_ptr, bool pred_guard = true)
{
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16), "Size is not supported");
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("{ \n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)pred_guard),
                 "r"(smem_int_ptr),
                 "l"(global_ptr),
                 "n"(SizeInBytes));
}

// only used for kernel pipeline analysis
template<int SizeInBytes>
__device__ __forceinline__ void cp_async_test_only(half* smem_ptr, const half* global_ptr, bool pred_guard = true)
{
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16), "Size is not supported");
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("{ \n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.cg.shared.global [%1], [%2], %3, 0;\n"
                 "}\n" ::"r"((int)pred_guard),
                 "r"(smem_int_ptr),
                 "l"(global_ptr),
                 "n"(SizeInBytes));
}

template<int SizeInBytes>
__device__ __forceinline__ void cp_async_ignore_src(half* smem_ptr, half* global_ptr)
{
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16), "Size is not supported");
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("{ \n"
                 "  cp.async.cg.shared.global [%0], [%1], %2, 0;\n"
                 "}\n" ::"r"(smem_int_ptr),
                 "l"(global_ptr),
                 "n"(SizeInBytes));
}

/// Establishes an ordering w.r.t previously issued cp.async instructions. Does not block.
__device__ __forceinline__ void cp_async_group_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

/// Blocks until all but <N> previous cp.async.commit_group operations have committed.
template<int N>
__device__ __forceinline__ void cp_async_wait_group()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

/// Blocks until all previous cp.async.commit_group operations have committed.
// cp.async.wait_all is equivalent to :
// cp.async.commit_group;
// cp.async.wait_group 0;
__device__ __forceinline__ void cp_async_wait_all()
{
    asm volatile("cp.async.wait_all;\n" ::);
}

template<int NumOfTensors>
__device__ __forceinline__ void FragLoadFromSharedToRegisters(uint32_t __restrict__ Registers[][4],
                                                              half* __restrict__ smem,
                                                              int warp_start_row,
                                                              int k_offset)
{
    //
    int lane_id = threadIdx.x % 32;
    int i       = lane_id % MMA_M;
    int j       = lane_id / MMA_M;
    //
    smem += TILE_K * (warp_start_row + i) + (k_offset + j * HALF_PER_128B);
    uint32_t __restrict__ smem_local_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    // Row Permutation to eliminating bank-conflict
    uint32_t RowLane_RowPermutation = i % COPY_UNIT_FP16_ROWS;
    uint32_t Mask_RowPermutation    = RowLane_RowPermutation << 4;
    smem_local_ptr                  = smem_local_ptr ^ Mask_RowPermutation;
//
#pragma unroll
    for (int i = 0; i < NumOfTensors; i++) {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(Registers[i][0]), "=r"(Registers[i][1]), "=r"(Registers[i][2]), "=r"(Registers[i][3])
                     : "r"(smem_local_ptr));

        smem_local_ptr += TILE_K * MMA_M * sizeof(half);
    }
}

template<int NumOfTensors, int N8>
__device__ __forceinline__ void B_FragLoadFromSharedToRegisters(uint32_t __restrict__ Registers[][4],
                                                                half* __restrict__ smem,
                                                                int warp_start_row,
                                                                int k_offset)
{
    //
    int      lane_id             = threadIdx.x % 32;
    int      i                   = lane_id % 8;
    uint32_t Mask_RowPermutation = i << 4;

    if (lane_id > 15)
        i += 8;
    int j = (lane_id % 16) >= 8 ? 1 : 0;
    //
    smem += TILE_K * (warp_start_row + i) + (k_offset + j * HALF_PER_128B);
    uint32_t __restrict__ smem_local_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    // Row Permutation to eliminating bank-conflict

    smem_local_ptr = smem_local_ptr ^ Mask_RowPermutation;
//
#pragma unroll
    for (int i = 0; i < NumOfTensors; i++) {
        // if(N8)
        //  asm volatile ("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        //              : "=r"(Registers[i][0]), "=r"(Registers[i][1])
        //              : "r"(smem_local_ptr));
        // else
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(Registers[i][0]), "=r"(Registers[i][1]), "=r"(Registers[i][2]), "=r"(Registers[i][3])
                     : "r"(smem_local_ptr));

        smem_local_ptr += TILE_K * MMA_N * sizeof(half);
    }
}

__device__ __forceinline__ void
MMA_FP16_M16N8K16(uint32_t __restrict__ c[], uint32_t __restrict__* a, uint32_t __restrict__* b)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                 "{ %0, %1, %2, %3},"
                 "{ %4, %5, %6, %7 },"
                 "{ %8, %9 },"
                 "{ %10, %11, %12, %13 };"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]),
                   "r"(a[1]),
                   "r"(a[2]),
                   "r"(a[3]),
                   "r"(b[0]),
                   "r"(b[1]),  /////////////// for column-major B
                   "r"(c[0]),
                   "r"(c[1]),
                   "r"(c[2]),
                   "r"(c[3]));
}

int cuda_CheckError()
{
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    return 0;
}

// New features: Copy size is X * 64, X can be any multiple to 8
template<int NumOfRowsToCopy, typename TilingConfig>  // NumOfRowsToCopy must be multiple to COPY_UNIT_FP16_ROWS
__device__ __forceinline__ void CopyTileFromGlobalToShared_X_64(half* __restrict__ SharedPTR,
                                                                const half* GlobalPTR,
                                                                const int   GlobalStride,
                                                                bool        Pred = true)
{
    //
    int lane_id       = threadIdx.x % 32;
    int col           = lane_id % 8;
    int row1          = lane_id / 8;
    int row2          = lane_id / 8 + 4;
    int store_column1 = col ^ row1;
    int store_column2 = col ^ row2;
    //
    int       warp_id            = threadIdx.x / 32;
    int       TotalNumOfCopyUnit = NumOfRowsToCopy / COPY_UNIT_FP16_ROWS;
    const int MaxIteration =
        (TotalNumOfCopyUnit - 1) / (TilingConfig::BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + 1;
//
#pragma unroll
    for (int i = 0; i < MaxIteration; i++) {
        int  COPY_UNIT_I        = (i * (TilingConfig::BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + warp_id);
        bool AsyncCopyPredictor = COPY_UNIT_I < TotalNumOfCopyUnit && Pred;  ///// Bug, too hard to find this bug, 5555
        const half* GlobalPTR_Unit        = GlobalPTR + COPY_UNIT_I * COPY_UNIT_FP16_ROWS * GlobalStride;
        half* __restrict__ SharedPTR_Unit = SharedPTR + COPY_UNIT_I * COPY_UNIT_FP16_ROWS * TILE_K;
        cp_async<16>(SharedPTR_Unit + store_column1 * HALF_PER_128B + row1 * TILE_K,
                     GlobalPTR_Unit + col * HALF_PER_128B + row1 * GlobalStride,
                     AsyncCopyPredictor);
        cp_async<16>(SharedPTR_Unit + store_column2 * HALF_PER_128B + row2 * TILE_K,
                     GlobalPTR_Unit + col * HALF_PER_128B + row2 * GlobalStride,
                     AsyncCopyPredictor);
        // cp_async_test_only<16>( SharedPTR_Unit + store_column1*HALF_PER_128B + row1 * TILE_K , GlobalPTR_Unit +
        // col*HALF_PER_128B + row1*GlobalStride, AsyncCopyPredictor ); cp_async_test_only<16>( SharedPTR_Unit +
        // store_column2*HALF_PER_128B + row2 * TILE_K , GlobalPTR_Unit + col*HALF_PER_128B + row2*GlobalStride,
        // AsyncCopyPredictor );
    }
}

template<typename TilingConfig>
__device__ __forceinline__ void PipelinedCoreComputations(float c[][REG_PER_C_TENSOR_16_16],
                                                          uint32_t __restrict__ a[][4],
                                                          uint32_t __restrict__ b[][4],
                                                          half* __restrict__ SharedMemoryPTR,
                                                          int warp_start_row,
                                                          int warp_start_col)
{
    uint32_t(*c_uint32_t)[REG_PER_C_TENSOR_16_16] = reinterpret_cast<uint32_t(*)[REG_PER_C_TENSOR_16_16]>(c);
    // First Register Loading
    FragLoadFromSharedToRegisters<WARP_ROW_TENSORS>(a, SharedMemoryPTR, warp_start_row, 0);
    B_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
        b, SharedMemoryPTR + TilingConfig::TILE_M * TILE_K, warp_start_col, 0);
// Sencond loading & first computation, so on
#pragma unroll
    for (int k = 0; k < BLOCK_K_TENSORS; k++) {
        uint32_t __restrict__(*a_read)[4]  = a;
        uint32_t __restrict__(*b_read)[4]  = b;
        uint32_t __restrict__(*a_write)[4] = a;
        uint32_t __restrict__(*b_write)[4] = b;
        a_read += ((k) % 2) * WARP_ROW_TENSORS;
        b_read += ((k) % 2) * TilingConfig::WARP_COL_TENSORS;
        a_write += ((k + 1) % 2) * WARP_ROW_TENSORS;
        b_write += ((k + 1) % 2) * TilingConfig::WARP_COL_TENSORS;
        // data loading
        if (k + 1 < BLOCK_K_TENSORS) {
            FragLoadFromSharedToRegisters<WARP_ROW_TENSORS>(a_write, SharedMemoryPTR, warp_start_row, (k + 1) * MMA_K);
            B_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
                b_write, SharedMemoryPTR + TilingConfig::TILE_M * TILE_K, warp_start_col, (k + 1) * MMA_K);
        }
// computations
#pragma unroll
        for (int i = 0; i < WARP_ROW_TENSORS; i++)
#pragma unroll
            for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) {
                // MMA_FP16_M16N16K16( c_uint32_t[i + j*WARP_ROW_TENSORS], a_read[i], b_read[j] );
                MMA_FP16_M16N8K16(c_uint32_t[i + j * WARP_ROW_TENSORS], a_read[i], b_read[j]);
                if (!TilingConfig::N8)
                    MMA_FP16_M16N8K16(c_uint32_t[i + j * WARP_ROW_TENSORS] + 4, a_read[i], b_read[j] + 2);  // c+4; b+2
            }
        //// only used for pipeline analysis
        //#pragma unroll
        // for (int i = 0; i < WARP_ROW_TENSORS; i++)
        //{
        //  int j=0;
        //  MMA_FP16_M16N8K16( c_uint32_t[i + j*WARP_ROW_TENSORS], a_read[i], b_read[j] );
        //}
        //#pragma unroll
        // for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++)
        //{
        //  int i=0;
        //  if(!TilingConfig::N8)
        //    MMA_FP16_M16N8K16( c_uint32_t[i + j*WARP_ROW_TENSORS]+4 , a_read[i], b_read[j]+2 );    // c+4; b+2
        //}
    }
}

template<typename TilingConfig>
__device__ __forceinline__ void
StoreToSharedMemoryFromRegister(float (*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C],
                                float c[][REG_PER_C_TENSOR_16_16])
{
    const unsigned int warpId        = threadIdx.x / WARP_SIZE;
    int                Warp_i        = warpId / TilingConfig::BLOCK_COL_WARPS;
    int                Warp_j        = warpId % TilingConfig::BLOCK_COL_WARPS;
    int                Warp_i_offset = Warp_i * (MMA_M * WARP_ROW_TENSORS);
    int                Warp_j_offset = Warp_j * (MMA_N * TilingConfig::WARP_COL_TENSORS);
    //
    int lane_id = threadIdx.x % WARP_SIZE;
//
#pragma unroll
    for (int i = 0; i < WARP_ROW_TENSORS; i++) {
#pragma unroll
        for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) {
            // Dealing with one 16*16 Tensor
            int RegSetID        = i + j * WARP_ROW_TENSORS;
            int Tensor_i_offset = Warp_i_offset + i * MMA_M;
            int Tensor_j_offset = Warp_j_offset + j * MMA_N;
#pragma unroll
            for (int r = 0; r < REG_PER_C_TENSOR_16_16; r++) {
                int row_offset = lane_id / 4;
                int col_offset = (lane_id % 4) * 2;
                //
                if (r % 2 > 0)
                    col_offset += 1;
                //
                if (r % 4 >= 2)
                    row_offset += 8;
                if (r >= 4)
                    col_offset += 8;
                //
                (*(smem_CFrag + Tensor_j_offset + col_offset))[Tensor_i_offset + row_offset] = c[RegSetID][r];
            }
        }
    }
}

#define ELEMENT_PER_THREADBLOCK 256

__global__ void SplitK_Reduction(half* C, half* Reduction_Workspace, int M_Global, int N_Global, int Split_K)
{
    // return;
    half* C_BasePTR_ThisBlock = C + ELEMENT_PER_THREADBLOCK * blockIdx.x;
    half* R_BasePTR_ThisBlock = Reduction_Workspace + ELEMENT_PER_THREADBLOCK * blockIdx.x;
    //
    float Results[HALF_PER_128B];
//
#pragma unroll
    for (int j = 0; j < HALF_PER_128B; j++)
        Results[j] = 0.0f;
    //
    for (int i = 0; i < Split_K; i++) {
#pragma unroll
        for (int j = 0; j < HALF_PER_128B; j++)
            Results[j] += __half2float(R_BasePTR_ThisBlock[threadIdx.x * HALF_PER_128B + j]);
        R_BasePTR_ThisBlock += M_Global * N_Global;
    }
#pragma unroll
    for (int j = 0; j < HALF_PER_128B; j++)
        C_BasePTR_ThisBlock[threadIdx.x * HALF_PER_128B + j] = __float2half_rn(Results[j]);
}

template<typename TilingConfig, typename SparseKernelConfig>
__device__ __forceinline__ void SpMM_CopyFromGlobalToReg(uint32_t*    Registers_GlobalToShared1,
                                                         uint32_t*    NNZ_VECTOR_ThreadLocal1,
                                                         const uint4* GlobalPTR1,
                                                         int          NNZ_VECTOR_ThisTile1,
                                                         uint32_t*    Registers_GlobalToShared2,
                                                         uint32_t*    NNZ_VECTOR_ThreadLocal2,
                                                         const uint4* GlobalPTR2,
                                                         int          NNZ_VECTOR_ThisTile2)
{
    // Load Global to registers
    int Num_NNZ_Vector1 = NNZ_VECTOR_ThisTile1 / (WARP_SIZE * TilingConfig::BLOCK_WARPS);
    if (threadIdx.x < (NNZ_VECTOR_ThisTile1 % (WARP_SIZE * TilingConfig::BLOCK_WARPS)))
        Num_NNZ_Vector1++;
    *NNZ_VECTOR_ThreadLocal1 = Num_NNZ_Vector1;
    if (TilingConfig::TILE_M == 256) {
        int Num_NNZ_Vector2 = NNZ_VECTOR_ThisTile2 / (WARP_SIZE * TilingConfig::BLOCK_WARPS);
        if (threadIdx.x < (NNZ_VECTOR_ThisTile2 % (WARP_SIZE * TilingConfig::BLOCK_WARPS)))
            Num_NNZ_Vector2++;
        *NNZ_VECTOR_ThreadLocal2 = Num_NNZ_Vector2;
    }
    //
    int Max_NNZ_VECTOR_ThisTile =
        (TilingConfig::TILE_M == 256) ? max(NNZ_VECTOR_ThisTile1, NNZ_VECTOR_ThisTile2) : NNZ_VECTOR_ThisTile1;
#pragma unroll
    for (int i = 0; i < SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / SparseKernelConfig::VECTOR_SIZE; i++) {
        int index = threadIdx.x + i * (WARP_SIZE * (TilingConfig::BLOCK_WARPS));
        if (index >= Max_NNZ_VECTOR_ThisTile)
            break;
        if (index < NNZ_VECTOR_ThisTile1
            || TilingConfig::TILE_M != 256)  // if TILE_M!=256, not need to compare since we have break();
        {
            Registers_GlobalToShared1[i * 4 + 0] = GlobalPTR1[index].x;
            Registers_GlobalToShared1[i * 4 + 1] = GlobalPTR1[index].y;
            Registers_GlobalToShared1[i * 4 + 2] = GlobalPTR1[index].z;
            Registers_GlobalToShared1[i * 4 + 3] = GlobalPTR1[index].w;
        }
        if (TilingConfig::TILE_M == 256)
            if (index < NNZ_VECTOR_ThisTile2) {
                Registers_GlobalToShared2[i * 4 + 0] = GlobalPTR2[index].x;
                Registers_GlobalToShared2[i * 4 + 1] = GlobalPTR2[index].y;
                Registers_GlobalToShared2[i * 4 + 2] = GlobalPTR2[index].z;
                Registers_GlobalToShared2[i * 4 + 3] = GlobalPTR2[index].w;
            }
    }
}

// Only used for kernel pipeline analysis, to make sure the global load for sparse encoding is not optimied by NVCC, we
// have to store the data loaded from GMem stored in SMem
template<typename TilingConfig, typename SparseKernelConfig>
__device__ __forceinline__ void SpMM_CopyFromGlobalToShared(int          tid,
                                                            half*        smem,
                                                            uint32_t*    Registers_GlobalToShared1,
                                                            uint32_t*    NNZ_VECTOR_ThreadLocal1,
                                                            const uint4* GlobalPTR1,
                                                            int          NNZ_VECTOR_ThisTile1,
                                                            uint32_t*    Registers_GlobalToShared2,
                                                            uint32_t*    NNZ_VECTOR_ThreadLocal2,
                                                            const uint4* GlobalPTR2,
                                                            int          NNZ_VECTOR_ThisTile2)
{
    uint32_t*    smem_int_ptr = reinterpret_cast<uint32_t*>(smem);
    unsigned int tmp1         = 0;
    unsigned int tmp2         = 0;
    // Load Global to registers
    int Num_NNZ_Vector1 = NNZ_VECTOR_ThisTile1 / (WARP_SIZE * TilingConfig::BLOCK_WARPS);
    if (threadIdx.x < (NNZ_VECTOR_ThisTile1 % (WARP_SIZE * TilingConfig::BLOCK_WARPS)))
        Num_NNZ_Vector1++;
    *NNZ_VECTOR_ThreadLocal1 = Num_NNZ_Vector1;
    if (TilingConfig::TILE_M == 256) {
        int Num_NNZ_Vector2 = NNZ_VECTOR_ThisTile2 / (WARP_SIZE * TilingConfig::BLOCK_WARPS);
        if (threadIdx.x < (NNZ_VECTOR_ThisTile2 % (WARP_SIZE * TilingConfig::BLOCK_WARPS)))
            Num_NNZ_Vector2++;
        *NNZ_VECTOR_ThreadLocal2 = Num_NNZ_Vector2;
    }
    //
    int Max_NNZ_VECTOR_ThisTile =
        (TilingConfig::TILE_M == 256) ? max(NNZ_VECTOR_ThisTile1, NNZ_VECTOR_ThisTile2) : NNZ_VECTOR_ThisTile1;
#pragma unroll
    for (int i = 0; i < SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / SparseKernelConfig::VECTOR_SIZE; i++) {
        int index = threadIdx.x + i * (WARP_SIZE * (TilingConfig::BLOCK_WARPS));
        if (index >= Max_NNZ_VECTOR_ThisTile)
            break;
        if (index < NNZ_VECTOR_ThisTile1
            || TilingConfig::TILE_M != 256)  // if TILE_M!=256, not need to compare since we have break();
        {
            tmp1 = GlobalPTR1[index].x + GlobalPTR1[index].y + GlobalPTR1[index].z + GlobalPTR1[index].w;
        }
        if (TilingConfig::TILE_M == 256)
            if (index < NNZ_VECTOR_ThisTile2) {
                tmp2 = GlobalPTR2[index].x + GlobalPTR2[index].y + GlobalPTR2[index].z + GlobalPTR2[index].w;
            }
    }
    smem_int_ptr[tid] = tmp1 + tmp2;
}

// Init Shared Memory to 0
template<typename TilingConfig>
__device__ __forceinline__ void SpMM_InitSharedMemory(half* __restrict__ SharedPTR)
{
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    //
    static_assert(TilingConfig::TILE_M % TilingConfig::BLOCK_WARPS == 0,
                  "TILE_M must be an integer multiple to BLOCK_WARPS");
    constexpr int RowsPerWarp = TilingConfig::TILE_M / TilingConfig::BLOCK_WARPS;
    //
    static_assert(TILE_K == 64, "For now, TILE_K is assumed to be 64.\n");
    const int StartRowNum         = warp_id * RowsPerWarp;
    half*     SharedPTR_PerThread = SharedPTR + StartRowNum * TILE_K + HALF_PER_128B * lane_id;
    //
    static_assert(RowsPerWarp % (WARP_SIZE * HALF_PER_128B / TILE_K) == 0,
                  "RowsPerWarp%(WARP_SIZE*HALF_PER_128B/TILE_K) should be 0\n");
    constexpr int ITERATIONS_PER_THREAD = RowsPerWarp / (WARP_SIZE * HALF_PER_128B / TILE_K);
#pragma unroll
    for (int i = 0; i < ITERATIONS_PER_THREAD; i++) {
        cp_async_ignore_src<16>(SharedPTR_PerThread, (half*)NULL);
        SharedPTR_PerThread += WARP_SIZE * HALF_PER_128B;
    }
}

template<typename TilingConfig, typename SparseKernelConfig>
__device__ __forceinline__ void SpMM_DecompressFromRegisterToShared(half* __restrict__ SharedPTR1,
                                                                    uint32_t* Registers_For_SparseTiles1,
                                                                    uint32_t  NNZ_ThreadLocal1,
                                                                    half* __restrict__ SharedPTR2,
                                                                    uint32_t* Registers_For_SparseTiles2,
                                                                    uint32_t  NNZ_ThreadLocal2)
{
    int Max_NNZ_ThreadLocal =
        (TilingConfig::TILE_M == 256) ? max(NNZ_ThreadLocal1, NNZ_ThreadLocal2) : NNZ_ThreadLocal1;
#pragma unroll
    for (int i = 0; i < SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / SparseKernelConfig::VECTOR_SIZE; i++) {
        if (i >= Max_NNZ_ThreadLocal)
            break;

        if (i < NNZ_ThreadLocal1
            || (TilingConfig::TILE_M != 256))  // if TILE_M!=256, not need to compare since we have break();
#pragma unroll
            for (int j = 0; j < SparseKernelConfig::VECTOR_SIZE; j++) {
                half* half_ptr =
                    reinterpret_cast<half*>(&(Registers_For_SparseTiles1[i * SparseKernelConfig::VECTOR_SIZE + j]));
                short* short_ptr  = reinterpret_cast<short*>(half_ptr + 1);
                half   value      = *half_ptr;
                short  index      = *short_ptr;
                SharedPTR1[index] = value;
            }

        if (TilingConfig::TILE_M == 256)
            if (i < NNZ_ThreadLocal2)
#pragma unroll
                for (int j = 0; j < SparseKernelConfig::VECTOR_SIZE; j++) {
                    half* half_ptr =
                        reinterpret_cast<half*>(&(Registers_For_SparseTiles2[i * SparseKernelConfig::VECTOR_SIZE + j]));
                    short* short_ptr  = reinterpret_cast<short*>(half_ptr + 1);
                    half   value      = *half_ptr;
                    short  index      = *short_ptr;
                    SharedPTR2[index] = value;
                }
    }
}


template<typename TilingConfig, typename SparseKernelConfig>
__global__ void SpMM_Kernel(const half*  A,
                            const uint4* Compressed_A,
                            const int*   TileOffsets,
                            const half*  B,
                            half*        Reduction_Workspace,
                            const int    M_Global,
                            const int    N_Global,
                            const int    K_Global,
                            int          Split_K)
{
    //
    const int BatchID     = blockIdx.y / (M_Global / TilingConfig::TILE_M);
    const int IsLastBatch = (BatchID == (Split_K - 1));
    const int x           = blockIdx.x;
    const int y           = blockIdx.y % (M_Global / TilingConfig::TILE_M);
    //
    const int NumKBlock        = K_Global / TILE_K;  // assert (K_Global%TILE_K==0);
    const int AverageNumKBlock = (NumKBlock - 1) / Split_K + 1;
    const int RoundedKBlock    = AverageNumKBlock * Split_K;
    const int PaddingKBlock    = RoundedKBlock - NumKBlock;
    int       NumIter          = 0;
    if (IsLastBatch)
        NumIter = AverageNumKBlock - PaddingKBlock;
    else
        NumIter = AverageNumKBlock;
    //
    const int* TileOffsets_ThisBlock1 = nullptr;
    const int* TileOffsets_ThisBlock2 = nullptr;
    if (TilingConfig::TILE_M == 256) {
        TileOffsets_ThisBlock1 =
            TileOffsets + K_Global / TILE_K * y * 2
            + BatchID * AverageNumKBlock;  // Address for matrix A, taking SplitK into consideration
        TileOffsets_ThisBlock2 =
            TileOffsets + K_Global / TILE_K * (y * 2 + 1)
            + BatchID * AverageNumKBlock;  // Address for matrix A, taking SplitK into consideration
    }
    else {
        TileOffsets_ThisBlock1 = TileOffsets + K_Global / TILE_K * y + BatchID * AverageNumKBlock;
        TileOffsets_ThisBlock2 = TileOffsets_ThisBlock1;  // otherwise will cause problem when passing
                                                          // TileOffsets_ThisBlock2[0] to SpMM_CopyFromGlobalToReg()
    }
    //
    uint32_t Registers_GlobalToShared[SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL];
    uint32_t NNZ_ThreadLocal1 = 0;
    uint32_t NNZ_ThreadLocal2 = 0;
    //
    extern __shared__ __align__(128) half smem[];  // at least be 128 Bytes aligned
    // Warp and lane identification.
    const unsigned int warpId       = threadIdx.x / WARP_SIZE;
    const int          Tile_Start_M = y * TilingConfig::TILE_M;
    const int          Tile_Start_N = x * TilingConfig::TILE_N;
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Compute a grid of C matrix tiles in each warp.
    int Warp_i         = warpId / TilingConfig::BLOCK_COL_WARPS;
    int Warp_j         = warpId % TilingConfig::BLOCK_COL_WARPS;
    int warp_start_row = WARP_ROW_TENSORS * MMA_M * Warp_i;
    int warp_start_col = TilingConfig::WARP_COL_TENSORS * MMA_N * Warp_j;
    uint32_t __restrict__ a[WARP_ROW_TENSORS * 2][4];
    uint32_t __restrict__ b[TilingConfig::WARP_COL_TENSORS * 2][4];
    // copying B tile from GlobalMemory to SharedMemory
    const half* BTileGlobalPTR =
        B + Tile_Start_N * K_Global
        + BatchID * AverageNumKBlock * TILE_K;  // Address for matrix B, taking SplitK into consideration
    //
    int NNZ_ThisTile1 = TileOffsets_ThisBlock1[1] - TileOffsets_ThisBlock1[0];
    int NNZ_ThisTile2 = 0;
    if (TilingConfig::TILE_M == 256)
        NNZ_ThisTile2 = TileOffsets_ThisBlock2[1] - TileOffsets_ThisBlock2[0];
    // printf("NNZ_ThisTile: %d ", NNZ_ThisTile);
    //
    SpMM_CopyFromGlobalToReg<TilingConfig, SparseKernelConfig>(Registers_GlobalToShared,
                                                               &NNZ_ThreadLocal1,
                                                               Compressed_A + TileOffsets_ThisBlock1[0],
                                                               NNZ_ThisTile1,
                                                               Registers_GlobalToShared
                                                                   + SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / 2,
                                                               &NNZ_ThreadLocal2,
                                                               Compressed_A + TileOffsets_ThisBlock2[0],
                                                               NNZ_ThisTile2);
    SpMM_InitSharedMemory<TilingConfig>(smem);
    cp_async_group_commit();
    CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(
        smem + TilingConfig::TILE_M * TILE_K, BTileGlobalPTR, K_Global);
    cp_async_group_commit();
    // Initilazing C Matrix to Zeros
    float c[WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS][REG_PER_C_TENSOR_16_16];
    for (int i = 0; i < WARP_ROW_TENSORS * TilingConfig::WARP_COL_TENSORS; i++)
        for (int j = 0; j < REG_PER_C_TENSOR_16_16; j++)
            c[i][j] = 0.0f;
    //
    cp_async_wait_group<1>();
    __syncthreads();
    SpMM_DecompressFromRegisterToShared<TilingConfig, SparseKernelConfig>(
        smem,
        Registers_GlobalToShared,
        NNZ_ThreadLocal1,
        smem + TilingConfig::TILE_M * TILE_K / 2,
        Registers_GlobalToShared + SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / 2,
        NNZ_ThreadLocal2);
    //
    cp_async_wait_group<0>();
    __syncthreads();
    // Prefetch to reduce stall_long_sb
    int StartIndex_SparseTiles_Prefetch1 = TileOffsets_ThisBlock1[0 + 1];
    int NNZ_ThisTile_Prefetch1           = TileOffsets_ThisBlock1[0 + 2] - TileOffsets_ThisBlock1[0 + 1];
    int StartIndex_SparseTiles_Prefetch2 = 0;
    int NNZ_ThisTile_Prefetch2           = 0;
    if (TilingConfig::TILE_M == 256) {
        StartIndex_SparseTiles_Prefetch2 = TileOffsets_ThisBlock2[0 + 1];
        NNZ_ThisTile_Prefetch2           = TileOffsets_ThisBlock2[0 + 2] - TileOffsets_ThisBlock2[0 + 1];
    }
// Debug
// printf("NNZ_ThisTile_Prefetch: %d ", NNZ_ThisTile_Prefetch);
//
// Go through the global K dimension by a fixed step at a time.
// write buffer[1] first, read buffer[0] first
#pragma unroll(1)
    for (int tile_id_k = 0; tile_id_k < NumIter; tile_id_k++) {
        // Using the previous prefetched value
        int StartIndex_SparseTiles1 = StartIndex_SparseTiles_Prefetch1;
        int NNZ_ThisTile1           = NNZ_ThisTile_Prefetch1;
        int StartIndex_SparseTiles2 = 0;
        int NNZ_ThisTile2           = 0;
        if (TilingConfig::TILE_M == 256) {
            StartIndex_SparseTiles2 = StartIndex_SparseTiles_Prefetch2;
            NNZ_ThisTile2           = NNZ_ThisTile_Prefetch2;
        }
        //
        StartIndex_SparseTiles_Prefetch1 = TileOffsets_ThisBlock1[tile_id_k + 1 + 1];
        NNZ_ThisTile_Prefetch1 = TileOffsets_ThisBlock1[tile_id_k + 1 + 2] - TileOffsets_ThisBlock1[tile_id_k + 1 + 1];
        if (TilingConfig::TILE_M == 256) {
            StartIndex_SparseTiles_Prefetch2 = TileOffsets_ThisBlock2[tile_id_k + 1 + 1];
            NNZ_ThisTile_Prefetch2 =
                TileOffsets_ThisBlock2[tile_id_k + 1 + 2] - TileOffsets_ThisBlock2[tile_id_k + 1 + 1];
        }
        // copying B tile from GlobalMemory to SharedMemory
        BTileGlobalPTR = B + Tile_Start_N * K_Global + BatchID * AverageNumKBlock * TILE_K + ((tile_id_k + 1) * TILE_K);
        // double buffer
        half* __restrict__ smem_write_PTR = smem;
        half* __restrict__ smem_read_PTR  = smem;
        smem_write_PTR = smem + ((tile_id_k + 1) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N);
        smem_read_PTR  = smem + ((tile_id_k) % 2) * (TilingConfig::TILE_M * TILE_K + TILE_K * TilingConfig::TILE_N);
        //
        bool GlobalCopy = (tile_id_k + 1) < NumIter;

        SpMM_InitSharedMemory<TilingConfig>(smem_write_PTR);
        cp_async_group_commit();
        SpMM_CopyFromGlobalToReg<TilingConfig, SparseKernelConfig>(
            Registers_GlobalToShared,
            &NNZ_ThreadLocal1,
            Compressed_A + StartIndex_SparseTiles1,
            NNZ_ThisTile1,
            Registers_GlobalToShared + SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / 2,
            &NNZ_ThreadLocal2,
            Compressed_A + StartIndex_SparseTiles2,
            NNZ_ThisTile2);

        // Copying B Tile
        CopyTileFromGlobalToShared_X_64<TilingConfig::TILE_N2, TilingConfig>(
            smem_write_PTR + TilingConfig::TILE_M * TILE_K, BTileGlobalPTR, K_Global, GlobalCopy);
        cp_async_group_commit();

        // only used for kernel pipeline analysis
        // SpMM_CopyFromGlobalToShared<TilingConfig, SparseKernelConfig>
        //               ( threadIdx.x,
        //                 smem_write_PTR,
        //                 Registers_GlobalToShared,
        //                 &NNZ_ThreadLocal1,
        //                 Compressed_A+StartIndex_SparseTiles1,
        //                 NNZ_ThisTile1,
        //                 Registers_GlobalToShared+SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL/2,
        //                 &NNZ_ThreadLocal2,
        //                 Compressed_A+StartIndex_SparseTiles2,
        //                 NNZ_ThisTile2);

        PipelinedCoreComputations<TilingConfig>(c, a, b, smem_read_PTR, warp_start_row, warp_start_col);
        //

        cp_async_wait_group<1>();
        __syncthreads();  // Sync to ensure the completion of stage 2, but the asyncopy of Tile_B may not finished yet
        SpMM_DecompressFromRegisterToShared<TilingConfig, SparseKernelConfig>(
            smem_write_PTR,
            Registers_GlobalToShared,
            NNZ_ThreadLocal1,
            smem_write_PTR + TilingConfig::TILE_M * TILE_K / 2,
            Registers_GlobalToShared + SparseKernelConfig::NUM_REG_FOR_SPARSE_KERNEL / 2,
            NNZ_ThreadLocal2);
        cp_async_wait_group<0>();  // Sync to ensure the completion of Loading B to shared memory
        __syncthreads();
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Store the C fragments to shared memory.
    float(*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C] =
        reinterpret_cast<float(*)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C]>(smem);
    StoreToSharedMemoryFromRegister<TilingConfig>(smem_CFrag, c);
    __syncthreads();
    // Now that shared memory contains all the D tiles, stream them to global memory.
    half* BlockGlobalPTR =
        Reduction_Workspace + BatchID * (M_Global * N_Global) + Tile_Start_M + Tile_Start_N * M_Global;
#pragma unroll
    for (int i = warpId; i < TilingConfig::TILE_N2; i += TilingConfig::BLOCK_WARPS)  // i-th column
#pragma unroll
        for (int j = threadIdx.x % WARP_SIZE; j < TilingConfig::TILE_M; j += WARP_SIZE)  // j-th row
            BlockGlobalPTR[j + i * M_Global] = __float2half_rn((*(smem_CFrag + i))[j]);
}

template<typename TilingConfig, typename SparseKernelConfig>
static void SpMM_SplitK_Kernel_Ex(cudaStream_t stream,
                                  const half*  A,
                                  const uint4* Compressed_A,
                                  const int*   TileOffsets,
                                  const half*  B,
                                  half*        Reduction_Workspace,
                                  const int    M_Global,
                                  const int    N_Global,
                                  const int    K_Global,
                                  int          Split_K)
{
    static int SHMEM_SZ = max((TilingConfig::TILE_M * TILE_K + TilingConfig::TILE_N * TILE_K) * sizeof(half) * 2,
                              (TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C) * TilingConfig::TILE_N * sizeof(float));
    cudaFuncSetAttribute(
        SpMM_Kernel<TilingConfig, SparseKernelConfig>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    // printf("Max shared memory size: %d B\n", SHMEM_SZ);
    int dimN =
        max(N_Global / TilingConfig::TILE_N, 1);  // max(N_Global/TilingConfig::TILE_N,1) used when N=8, TILE_N=16
    int  dimM = M_Global * Split_K / TilingConfig::TILE_M;
    dim3 GridDim(dimN, dimM, 1);  // Grid Size is increased due to SplitK for higher SM occupancy
    dim3 BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS, 1, 1);
    //
    SpMM_Kernel<TilingConfig, SparseKernelConfig><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
        A, Compressed_A, TileOffsets, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);
}

/*
half* Reduction_Workspace:  1. Requiring an extra memory space in device memory for un-reducted intermediate output
tensors
                            2. Reduction_Workspace_Size = max( Split_K * M_Global * N_Global ) * sizeof(fp16)
int Split_K:                Split K dimension into Split_K Parts
*/
cudaError_t SpMM_SplitK_API(cudaStream_t stream,
                            const half*  A,
                            const uint4* Compressed_A,
                            const int*   TileOffsets,
                            const half*  B,
                            half*        C,
                            const int    M_Global,
                            const int    N_Global,
                            const int    K_Global,
                            half*        Reduction_Workspace,  // Identical workspace for all SpMM kernel launches
                            int          Split_K)
{
#ifdef DEBUG_MODE
    printf(
        "SpMM_API.cu->SpMM_SplitK_API():  M: %d, N: %d, K: %d, SplitK: %d \n", M_Global, N_Global, K_Global, Split_K);
    assert(K_Global % TILE_K == 0);
    assert(M_Global % 256 == 0);
#endif
    half* SpMM_SplitK_OutputPTR;
    if (Split_K == 1)
        SpMM_SplitK_OutputPTR = C;
    else
        SpMM_SplitK_OutputPTR = Reduction_Workspace;
    // Batched SpMM
    switch (N_Global) {
        case 8:
            SpMM_SplitK_Kernel_Ex<TilingConfig<4, 1, 1, 1>, SparseKernelConfig<96>>(
                stream, A, Compressed_A, TileOffsets, B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        case 16:
            SpMM_SplitK_Kernel_Ex<TilingConfig<4, 1, 1>, SparseKernelConfig<96>>(
                stream, A, Compressed_A, TileOffsets, B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        case 32:
            SpMM_SplitK_Kernel_Ex<TilingConfig<4, 1, 2>, SparseKernelConfig<96>>(
                stream, A, Compressed_A, TileOffsets, B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        case 64:
            // return SpMM_SplitK_Kernel_Ex< TilingConfig<4, 1, 4>, SparseKernelConfig<64> >
            SpMM_SplitK_Kernel_Ex<TilingConfig<2, 2, 2>, SparseKernelConfig<64>>(
                stream, A, Compressed_A, TileOffsets, B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        case 128:
            SpMM_SplitK_Kernel_Ex<TilingConfig<2, 2, 4>, SparseKernelConfig<64>>(
                stream, A, Compressed_A, TileOffsets, B, SpMM_SplitK_OutputPTR, M_Global, N_Global, K_Global, Split_K);
            break;
        default:
            if (N_Global % 128 == 0)
                SpMM_SplitK_Kernel_Ex<TilingConfig<2, 2, 4>, SparseKernelConfig<64>>(stream,
                                                                                     A,
                                                                                     Compressed_A,
                                                                                     TileOffsets,
                                                                                     B,
                                                                                     SpMM_SplitK_OutputPTR,
                                                                                     M_Global,
                                                                                     N_Global,
                                                                                     K_Global,
                                                                                     Split_K);
            else {
                printf("MM_Sparse_API Error: Unsupported N dimension %d!\n", N_Global);
                return cudaErrorUnknown;
            }
            break;
    }
    //
    cudaError_t Error = cudaGetLastError();
    if (Error != cudaSuccess)
        return Error;

    if (Split_K == 1)
        return Error;
    dim3 GridDim((M_Global * N_Global) / 256, 1, 1);
    dim3 BlockDim(WARP_SIZE, 1, 1);
    SplitK_Reduction<<<GridDim, BlockDim, 0, stream>>>(C, Reduction_Workspace, M_Global, N_Global, Split_K);
    return cudaGetLastError();
}

static int BankID_Minimum(std::vector<unsigned int> ItemsInBank[])
{
    int ID           = 0;
    int MinItemCount = ItemsInBank[0].size();
    for (int i = 1; i < 32; i++) {
        if (ItemsInBank[i].size() < MinItemCount) {
            ID           = i;
            MinItemCount = ItemsInBank[i].size();
        }
    }
    return ID;
}

static int BankID_Maximum(std::vector<unsigned int> ItemsInBank[])
{
    int ID           = 0;
    int MaxItemCount = ItemsInBank[0].size();
    for (int i = 1; i < 32; i++) {
        if (ItemsInBank[i].size() > MaxItemCount) {
            ID           = i;
            MaxItemCount = ItemsInBank[i].size();
        }
    }
    return ID;
}

/*
return: Number of Element in array TileOffsets
Note: TileOffsets[return-1] = NNZ / SparseKernelConfig::VECTOR_SIZE    (SparseKernelConfig::VECTOR_SIZE = 4)
*/
// template<typename TilingConfig, typename SparseKernelConfig>
__host__ int InitSparseMatrixA_API(half*      A_h,
                                   int        M,
                                   int        N,
                                   int        K,
                                   uint32_t** Compressed_A,  // CPU PTR
                                   int**      TileOffsets)        // CPU_PTR
{
    // Unified Sparse Fornat for different N, in our kernel, TILE_M=128 or 256
    const int TILE_M                       = 128;
    const int VECTOR_SIZE                  = 4;
    const int PADDING_SIZE_FOR_TILEOFFSETS = 2;
#ifdef DEBUG_MODE
    printf("Weight Shuffle is Enabled\n");
#endif
    float ZERO_THRESHOLD = 0.0;
    int   NumRow_offsets = M / TILE_M;
    int   NumCol_offsets = K / TILE_K;
    //
    int NNZ_Original = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            if (fabs(__half2float(A_h[i * K + j])) > ZERO_THRESHOLD)
                NNZ_Original++;
#ifdef DEBUG_MODE
    printf("Matrix A: M=%d K=%d, NNZ=%d, Pruning Ratio=%.2f\n",
           M,
           K,
           NNZ_Original,
           1.0f - static_cast<float>(NNZ_Original) / (M * K));
#endif
    //
    int  NNZ_AfterPadding   = 0;
    int* PaddingForEachTile = (int*)malloc(NumRow_offsets * NumCol_offsets * sizeof(int));
    if (!PaddingForEachTile) {
        printf("Error in InitSparseMatrixA line %d :malloc Error\n", __LINE__);
        exit(-1);
    }
    for (int i = 0; i < M / TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half* CurrentTilePTR = A_h + (i * TILE_M) * K + (j * TILE_K);
            int   TileNZCount    = 0;
            for (int m = 0; m < TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD)
                        TileNZCount++;
                }
            }
            int NumPadding                           = (VECTOR_SIZE - (TileNZCount % VECTOR_SIZE)) % VECTOR_SIZE;
            PaddingForEachTile[i * (K / TILE_K) + j] = NumPadding;
            TileNZCount += NumPadding;
            NNZ_AfterPadding += TileNZCount;
        }
    }
#ifdef DEBUG_MODE
    printf("Matrix A: M=%d K=%d, NNZ_AfterPadding=%d, PruningRatio_AfterPadding=%.2f\n",
           M,
           K,
           NNZ_AfterPadding,
           1.0f - static_cast<float>(NNZ_AfterPadding) / (M * K));
#endif
    //
    *Compressed_A = (uint32_t*)malloc(NNZ_AfterPadding * sizeof(uint32_t));
    *TileOffsets  = (int*)malloc((NumRow_offsets * NumCol_offsets + PADDING_SIZE_FOR_TILEOFFSETS) * sizeof(int));
    if (*Compressed_A == NULL || *TileOffsets == NULL) {
        printf("InitSparseMatrixA: Error in malloc memory from host memory!\n");
        exit(-1);
    }
    // Generating compressed format for A Matrix
    assert(M % TILE_M == 0 && K % TILE_K == 0);
    int       TotalNZCount = 0;
    uint32_t* Ptr_SubArray = *Compressed_A;
    for (int i = 0; i < M / TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half*        CurrentTilePTR    = A_h + (i * TILE_M) * K + (j * TILE_K);
            int          TileNZCount       = 0;
            int          remainingPaddings = PaddingForEachTile[i * (K / TILE_K) + j];
            unsigned int Item              = 0;
            // Processing each tile
            std::vector<unsigned int> ItemsInBank[32];
            int                       ZeroPositionForBank[32];
            for (int k = 0; k < 32; k++)
                ZeroPositionForBank[k] = -1;
            //
            // printf("Starting Processing Tile i:%d j:%d...\n", i, j);
            for (int m = 0; m < TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    // Row permutation for bank-conflict-free shared memory layout
                    int      row            = m;
                    int      col            = n;
                    uint32_t mask           = (row % 8) << 3;
                    int      col_permutated = col ^ mask;
                    int      bank_smem      = (col_permutated / 2) % 32;
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD) {
                        half* half_ptr   = reinterpret_cast<half*>(&Item);
                        *half_ptr        = value;
                        short* short_ptr = reinterpret_cast<short*>(half_ptr + 1);
                        *short_ptr       = static_cast<short>(row * TILE_K + col_permutated);
                        ItemsInBank[bank_smem].push_back(Item);
                        //
                        TileNZCount++;
                    }
                    else {
                        if (ZeroPositionForBank[bank_smem] == -1)
                            ZeroPositionForBank[bank_smem] = row * TILE_K + col_permutated;
                    }
                }
            }
            //
            // printf("Starting Weight Padding...\n");
            for (int k = 0; k < remainingPaddings; k++) {
                int BankID = BankID_Minimum(ItemsInBank);
                assert(BankID >= 0 && BankID < 32);
                int ZeroPosition = ZeroPositionForBank[BankID];
                assert(ZeroPosition != -1);
                //
                half* half_ptr   = reinterpret_cast<half*>(&Item);
                *half_ptr        = __float2half_rn(0.0f);
                short* short_ptr = reinterpret_cast<short*>(half_ptr + 1);
                *short_ptr       = static_cast<short>(ZeroPosition);
                ItemsInBank[BankID].push_back(Item);
                //
                TileNZCount++;
            }
            /*
            if(i==0 && j==0)
            {
              printf("For tile i:%d j:%d\n",i,j);
              for(int h=0; h<32; h++)
                printf("%ld ", ItemsInBank[h].size());
              printf("\n");
            }
            */
            //
            // printf("Starting Weight Shuffle...\n");
            std::vector<unsigned int> MainPart[32];
            std::vector<unsigned int> TailPart[32];
            int                       TileVectorCount = TileNZCount / VECTOR_SIZE;
            assert(TileNZCount % VECTOR_SIZE == 0);
            int Repeat_Vector   = TileVectorCount / WARP_SIZE;
            int Remained_Vector = TileVectorCount % WARP_SIZE;
            // Filing the TailPart
            for (int v = 0; v < VECTOR_SIZE; v++) {
                for (int b = 0; b < Remained_Vector; b++) {
                    int BankID = BankID_Maximum(ItemsInBank);
                    Item       = ItemsInBank[BankID].back();
                    ItemsInBank[BankID].pop_back();
                    TailPart[b].push_back(Item);
                }
            }
            // Filing the MainPart
            // printf("Starting Filing the MainPart...\n");
            for (int r = 0; r < Repeat_Vector; r++) {
                for (int v = 0; v < VECTOR_SIZE; v++) {
                    for (int b = 0; b < WARP_SIZE; b++) {
                        int BankID = BankID_Maximum(ItemsInBank);
                        Item       = ItemsInBank[BankID].back();
                        ItemsInBank[BankID].pop_back();
                        MainPart[b].push_back(Item);
                    }
                }
            }
            // Writing to the Sub-Array
            // printf("Starting Writing to the Sub-Array...\n");
            for (int r = 0; r < Repeat_Vector; r++) {
                for (int v = 0; v < VECTOR_SIZE; v++) {
                    for (int b = 0; b < 32; b++) {
                        Item = MainPart[b].back();
                        MainPart[b].pop_back();
                        int V_Size                                     = VECTOR_SIZE;
                        Ptr_SubArray[r * V_Size * 32 + b * V_Size + v] = Item;
                    }
                }
            }
            Ptr_SubArray += Repeat_Vector * VECTOR_SIZE * WARP_SIZE;
            for (int v = 0; v < VECTOR_SIZE; v++) {
                for (int b = 0; b < Remained_Vector; b++) {
                    Item = TailPart[b].back();
                    TailPart[b].pop_back();
                    Ptr_SubArray[b * VECTOR_SIZE + v] = Item;
                }
            }
            Ptr_SubArray += VECTOR_SIZE * Remained_Vector;
            //
            TotalNZCount += TileNZCount;
            (*TileOffsets)[i * K / TILE_K + j + 1] = TotalNZCount / VECTOR_SIZE;
        }
    }
    //
    assert(TotalNZCount == NNZ_AfterPadding);
    (*TileOffsets)[0] = 0;
    (*TileOffsets)[(M / TILE_M) * (K / TILE_K) + 1] =
        TotalNZCount / VECTOR_SIZE;  // #define PADDING_SIZE_FOR_TILEOFFSETS 2  // (N+1 offsets) + 1 padding // adding
                                     // an empty tile at last
    //
    return (M / TILE_M) * (K / TILE_K) + 2;  // number of Elements in array TileOffsets
}

// A_h is host memory pointer, Compressed_A and TileOffsets are device memory pointers
__host__ int InitSparseMatrixA_API_NoReorder(half*      A_h,
                                             int        M,
                                             int        N,
                                             int        K,
                                             uint32_t** Compressed_A,  // CPU PTR
                                             int**      TileOffsets)        // CPU_PTR
{
    // Unified Sparse Fornat for different N, in our kernel, TILE_M=128 or 256
    const int TILE_M                       = 128;
    const int VECTOR_SIZE                  = 4;
    const int PADDING_SIZE_FOR_TILEOFFSETS = 2;
#ifdef DEBUG_MODE
    printf("Weight Shuffle is NOT Enabled\n");
#endif
    float ZERO_THRESHOLD = 0.0;
    int   NumRow_offsets = M / TILE_M;
    int   NumCol_offsets = K / TILE_K;
    //
    int NNZ_Original = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            if (fabs(__half2float(A_h[i * K + j])) > ZERO_THRESHOLD)
                NNZ_Original++;
#ifdef DEBUG_MODE
    printf("Matrix A: M=%d K=%d, NNZ=%d, Pruning Ratio=%.2f\n",
           M,
           K,
           NNZ_Original,
           1.0f - static_cast<float>(NNZ_Original) / (M * K));
#endif
    //
    int  NNZ_AfterPadding   = 0;
    int* PaddingForEachTile = (int*)malloc(NumRow_offsets * NumCol_offsets * sizeof(int));
    if (!PaddingForEachTile) {
        printf("Error in InitSparseMatrixA line %d :malloc Error\n", __LINE__);
        exit(-1);
    }
    for (int i = 0; i < M / TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half* CurrentTilePTR = A_h + (i * TILE_M) * K + (j * TILE_K);
            int   TileNZCount    = 0;
            for (int m = 0; m < TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD)
                        TileNZCount++;
                }
            }
            int NumPadding                           = (VECTOR_SIZE - (TileNZCount % VECTOR_SIZE)) % VECTOR_SIZE;
            PaddingForEachTile[i * (K / TILE_K) + j] = NumPadding;
            TileNZCount += NumPadding;
            NNZ_AfterPadding += TileNZCount;
        }
    }
#ifdef DEBUG_MODE
    printf("Matrix A: M=%d K=%d, NNZ_AfterPadding=%d, PruningRatio_AfterPadding=%.2f\n",
           M,
           K,
           NNZ_AfterPadding,
           1.0f - static_cast<float>(NNZ_AfterPadding) / (M * K));
#endif
    //
    *Compressed_A = (uint32_t*)malloc(NNZ_AfterPadding * sizeof(uint32_t));
    *TileOffsets  = (int*)malloc((NumRow_offsets * NumCol_offsets + PADDING_SIZE_FOR_TILEOFFSETS) * sizeof(int));
    if (*Compressed_A == NULL || *TileOffsets == NULL) {
        printf("InitSparseMatrixA: Error in malloc memory from host memory!\n");
        exit(-1);
    }
    // Generating compressed format for A Matrix
    assert(M % TILE_M == 0 && K % TILE_K == 0);
    int TotalNZCount = 0;
    for (int i = 0; i < M / TILE_M; i++) {
        for (int j = 0; j < K / TILE_K; j++) {
            half* CurrentTilePTR    = A_h + (i * TILE_M) * K + (j * TILE_K);
            int   TileNZCount       = 0;
            int   remainingPaddings = PaddingForEachTile[i * (K / TILE_K) + j];
            for (int m = 0; m < TILE_M; m++) {
                for (int n = 0; n < TILE_K; n++) {
                    half value = CurrentTilePTR[m * K + n];
                    if (fabs(__half2float(value)) > ZERO_THRESHOLD) {
                        half* half_ptr   = reinterpret_cast<half*>(*Compressed_A + TotalNZCount);
                        *half_ptr        = value;
                        short* short_ptr = reinterpret_cast<short*>(half_ptr + 1);
                        // Row permutation for bank-conflict-free shared memory layout
                        int      row            = m;
                        int      col            = n;
                        uint32_t mask           = (row % 8) << 3;
                        int      col_permutated = col ^ mask;
                        *short_ptr              = static_cast<short>(row * TILE_K + col_permutated);
                        //
                        TileNZCount++;
                        TotalNZCount++;
                    }
                    else {
                        if (remainingPaddings > 0) {
                            remainingPaddings--;
                            half* half_ptr   = reinterpret_cast<half*>(*Compressed_A + TotalNZCount);
                            *half_ptr        = value;  // zero
                            short* short_ptr = reinterpret_cast<short*>(half_ptr + 1);
                            // Row permutation for bank-conflict-free shared memory layout
                            int      row            = m;
                            int      col            = n;
                            uint32_t mask           = (row % 8) << 3;
                            int      col_permutated = col ^ mask;
                            *short_ptr              = static_cast<short>(row * TILE_K + col_permutated);
                            //
                            TileNZCount++;
                            TotalNZCount++;
                        }
                    }
                }
            }
            //
            assert(TileNZCount % VECTOR_SIZE == 0);
            (*TileOffsets)[i * K / TILE_K + j + 1] = TotalNZCount / VECTOR_SIZE;
        }
    }
    assert(TotalNZCount == NNZ_AfterPadding);
    (*TileOffsets)[0] = 0;
    (*TileOffsets)[(M / TILE_M) * (K / TILE_K) + 1] =
        TotalNZCount / VECTOR_SIZE;  // #define PADDING_SIZE_FOR_TILEOFFSETS 2  // (N+1 offsets) + 1 padding // adding
                                     // an empty tile at last
    //

    //
    return (M / TILE_M) * (K / TILE_K) + 2;  // number of Elements in array TileOffsets
}

/*
input:    char* DenseMatrixFileName
          int   M
          int   N                   // N is used by void InitSparseMatrixA_API()
          int   K
          char* NZWeightsFileName
          char* TileOffsetsFileName
          char* OutputSizesFileName // NNZ -> NumOffsets
*/
extern "C" void GenSparseMatrixBinFile(char* DenseMatrixFileName,
                                       int   M,
                                       int   K,
                                       char* NZWeightsFileName,
                                       char* TileOffsetsFileName,
                                       char* OutputSizesFileName)
{
    std::vector<half> host_array(M * K);
    std::ifstream     in(DenseMatrixFileName, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        printf("file %s cannot be opened, loadDataArrayFromBin fails. \n", DenseMatrixFileName);
        exit(-1);
    }
    size_t loaded_data_size = sizeof(half) * M * K;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);
#ifdef DEBUG_MODE
    printf("Read %ld bytes from %s.\n", loaded_data_size, DenseMatrixFileName);
#endif
    in.read((char*)host_array.data(), loaded_data_size);
    size_t in_get_size = in.gcount();
    if (in_get_size != loaded_data_size) {
        printf("file %s only has %ld, but request %ld, loading DenseMatrix fails! \n",
               DenseMatrixFileName,
               in_get_size,
               loaded_data_size);
        exit(-1);
    }
    in.close();
    // Step 2: Dense to Sparse Transformation
    unsigned int* NZWeights_CPU   = nullptr;
    int*          TileOffsets_CPU = nullptr;
    int           NumOffsets      = InitSparseMatrixA_API(host_array.data(), M, 0, K, &NZWeights_CPU, &TileOffsets_CPU);
    int           NNZ             = TileOffsets_CPU[NumOffsets - 1] * 4;  // VectorSize = 4
    // Step 3: Write to FILE(OutputSizesFileName)
    //         Write to FILE(NZWeightsFileName), FILE(TileOffsetsFileName)
    std::ofstream out_SizesFile(OutputSizesFileName, std::ios::out | std::ios::binary);
    std::ofstream out_NZWeightsFile(NZWeightsFileName, std::ios::out | std::ios::binary);
    std::ofstream out_TileOffsetsFile(TileOffsetsFileName, std::ios::out | std::ios::binary);
    if (!out_SizesFile.is_open() || !out_NZWeightsFile.is_open() || !out_TileOffsetsFile.is_open()) {
        printf("GenSparseMatrixBinFile() ERROR: file %s, %s, or %s cannot be opened or creaetd. \n",
               OutputSizesFileName,
               NZWeightsFileName,
               TileOffsetsFileName);
        exit(-1);
    }
    //
    // out_SizesFile << NNZ << NumOffsets;
    out_SizesFile.write((char*)&NNZ, sizeof(int));
    out_SizesFile.write((char*)&NumOffsets, sizeof(int));
    out_SizesFile.close();
    out_NZWeightsFile.write((char*)NZWeights_CPU, sizeof(uint32_t) * NNZ);
    out_NZWeightsFile.close();
    out_TileOffsetsFile.write((char*)TileOffsets_CPU, sizeof(int) * NumOffsets);
    out_TileOffsetsFile.close();
}


}


#endif