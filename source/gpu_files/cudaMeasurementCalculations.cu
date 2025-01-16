#pragma once

#include <curand.h>
#include "c_headers.hpp"
#include "tensor.cuh"
#include "real_type.h"
#include "cudaStructures.hpp"
#include "CudaMeasurementCalculations.cuh"
#include <numeric>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__inline__ __device__
real warpReduceSum(real val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down(val, offset);
    return val;
}

__global__ void GPUMagnSum(const CudaTensor<real, 3> spin, CudaTensor<real, 2> mblock, int tasks) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    int lane = warp.thread_rank();
    int wid = warp.meta_group_rank();
    int wSize = warp.size();
    int wNum = warp.meta_group_size();
    int tid = grid.thread_rank();
    int tid_in_block = block.thread_rank();

    int mInd = grid.block_index().y;
    int offsetM = mInd * tasks;
    int tid_in_M = grid.block_index().x * block.num_threads() + tid_in_block;
    int stride = grid.dim_blocks().x * block.num_threads();

    real mySum[3] = { 0.0, 0.0, 0.0 };
    static __shared__ real shared0[32];
    static __shared__ real shared1[32];
    static __shared__ real shared2[32];


    for (int id = tid_in_M; id < tasks; id += stride) {
        mySum[id % 3] += spin[id + offsetM];
        //  printf("tid = %i, mInd = %i, stride = %i, data_id = %i, mySum = %.3f\n", tid, mInd , stride, id + offsetM, mySum[id % 3]);
    }
    warp.sync();

    mySum[0] = warpReduceSum(mySum[0]);
    mySum[1] = warpReduceSum(mySum[1]);
    mySum[2] = warpReduceSum(mySum[2]);

    if (lane == 0) {
        shared0[wid] = mySum[0];
        shared1[wid] = mySum[1];
        shared2[wid] = mySum[2];
    }

    __syncthreads();              // Wait for all partial reductions
    mySum[0] = (tid_in_block < wNum) ? shared0[lane] : 0;
    mySum[1] = (tid_in_block < wNum) ? shared1[lane] : 0;
    mySum[2] = (tid_in_block < wNum) ? shared2[lane] : 0;

    if (wid == 0) {
        mySum[0] = warpReduceSum(mySum[0]); //Final reduce within first warp
        mySum[1] = warpReduceSum(mySum[1]);
        mySum[2] = warpReduceSum(mySum[2]);
    }

    if (tid_in_block == 0) {
        mblock(block.group_index().x, mInd) = mySum[0];
        mblock(block.group_index().x + grid.group_dim().x, mInd) = mySum[1];
        mblock(block.group_index().x + 2 * grid.group_dim().x, mInd) = mySum[2];
        // printf("mInd = %i, bid = %i, mblock0 = %.3f\n", mInd, block.group_index().x, mblock(block.group_index().x, mInd));
         //printf("tid = %i, mInd = %i, mblock0 = %lf, mblock1 = %lf, mblock2 = %lf\n", tid, mInd, mblock(block.group_index().x, mInd), mblock(block.group_index().x + grid.group_dim().x, mInd), mblock(block.group_index().x + 2 * grid.group_dim().x, mInd));
    }
}
__global__ void GPUMagnFinalSum(CudaTensor<real, 2> mblock, CudaTensor<real, 3> msum, int numBlocks, int curstep)
{
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    int lane = warp.thread_rank();
    int wid = warp.meta_group_rank();
    int wSize = warp.size();
    int wNum = warp.meta_group_size();
    int tid = grid.thread_rank();
    int tNum = block.size();
    int tid_in_block = block.thread_rank();

    int mInd = grid.block_index().x;
    int offsetM = mInd * numBlocks;
    int tid_in_M = tid_in_block;
    //printf("numblocks = %i\n", numBlocks);

    real mySum[3] = { 0.0, 0.0, 0.0 };
    static __shared__ real shared0[32];
    static __shared__ real shared1[32];
    static __shared__ real shared2[32];

    if (tid_in_M < numBlocks) {
        mySum[0] += mblock(tid_in_M, mInd);
        mySum[1] += mblock(tid_in_M + numBlocks, mInd);
        mySum[2] += mblock(tid_in_M + 2 * numBlocks, mInd);
        //printf("tid_in_m = %i, mInd = %i, mblock = %.3f\n", tid_in_M, mInd, mblock(tid_in_M, mInd));
    }

    warp.sync();

    mySum[0] = warpReduceSum(mySum[0]); //Final reduce within first warp
    mySum[1] = warpReduceSum(mySum[1]); //Final reduce within first warp
    mySum[2] = warpReduceSum(mySum[2]); //Final reduce within first warp

    if (lane == 0) {
        shared0[wid] = mySum[0];
        shared1[wid] = mySum[1];
        shared2[wid] = mySum[2];
    }

    __syncthreads();              // Wait for all partial reductions
    mySum[0] = (tid_in_block < wNum) ? shared0[lane] : 0;
    mySum[1] = (tid_in_block < wNum) ? shared1[lane] : 0;
    mySum[2] = (tid_in_block < wNum) ? shared2[lane] : 0;
    if (wid == 0) mySum[0] = warpReduceSum(mySum[0]); //Final reduce within first warp
    if (wid == 0) mySum[1] = warpReduceSum(mySum[1]); //Final reduce within first warp
    if (wid == 0) mySum[2] = warpReduceSum(mySum[2]); //Final reduce within first warp

    if (tid_in_block == 0) {
        msum(0, curstep, mInd) = mySum[0];
        msum(1, curstep, mInd) = mySum[1];
        msum(2, curstep, mInd) = mySum[2];
        /*mblock_gpu[block.group_index().x] += mySum[0];
        mblock_gpu[block.group_index().x + grid.group_dim().x] += mySum[1];
        mblock_gpu[block.group_index().x + 2 * grid.group_dim().x] += mySum[2];*/
        printf("mInd = %i, mblock0 = %lf, mblock1 = %lf, mblock2 = %lf\n", mInd, msum(0, curstep, mInd), msum(1, curstep, mInd), msum(2, curstep, mInd));
    }
}

__global__ void GPUEnergySum(const CudaTensor<real, 3> spin, const CudaTensor<real, 3> beff, CudaTensor<real, 2> eblock, int tasks) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    int lane = warp.thread_rank();
    int wid = warp.meta_group_rank();
    int wSize = warp.size();
    int wNum = warp.meta_group_size();
    int tid = grid.thread_rank();
    int tid_in_block = block.thread_rank();

    int mInd = grid.block_index().y;
    int offsetM = mInd * tasks;
    int tid_in_M = grid.block_index().x * block.num_threads() + tid_in_block;
    int stride = grid.dim_blocks().x * block.num_threads();

    real mySum = 0.0;
    //real curSum = 0.0;
    static __shared__ real shared0[32];

    for (int id = tid_in_M; id < tasks; id += stride) {
        mySum += spin[id + offsetM] * beff[id + offsetM];
        //curSum = spin[id + offsetM]*beff[id + offsetM];
       //printf("tid = %i, site = %i, comp = %i, beff = %lf, spin = %lf, curE = %lf, sumE = %lf\n", id, id/3, id%3, beff[id + offsetM], spin[id + offsetM], curSum, mySum);
    }
    warp.sync();
    //if (tid < tasks) printf("before warp, tid = %i, sumE = %lf\n", tid, mySum);
    mySum = warpReduceSum(mySum); //Final reduce within first warp
    //if (tid < tasks) printf("after warp, tid = %i, sumE = %lf\n", tid, mySum);
    if (lane == 0) {
        shared0[wid] = mySum;
    }
    __syncthreads();              // Wait for all partial reductions
    mySum = (tid_in_block < wNum) ? shared0[lane] : 0;
    if (wid == 0) mySum = warpReduceSum(mySum); //Final reduce within first warp
    if (tid_in_block == 0) {
         eblock(block.group_index().x, mInd) = mySum;
        // printf("mInd = %i, block = %i,eblock = %lf\n", mInd, block.group_index().x, eblock(block.group_index().x, mInd));
    }
}

__global__ void GPUScalarFinalSum(CudaTensor<real, 2> block_value, CudaTensor<real, 2> scalar_sum, int numBlocks, int curstep)
{
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    int lane = warp.thread_rank();
    int wid = warp.meta_group_rank();
    int wSize = warp.size();
    int wNum = warp.meta_group_size();
    int tid = grid.thread_rank();
    int tNum = block.size();
    int tid_in_block = block.thread_rank();

    int mInd = grid.block_index().x;
    int offsetM = mInd * numBlocks;
    int tid_in_M = tid_in_block;

    real mySum = 0.0;
    static __shared__ real shared0[32];

    if (tid_in_M < numBlocks) { mySum += block_value(tid_in_M, mInd); }

    warp.sync();
    mySum = warpReduceSum(mySum); //Final reduce within first warp
    if (lane == 0) { shared0[wid] = mySum; }

    __syncthreads();              // Wait for all partial reductions

    mySum = (tid_in_block < wNum) ? shared0[lane] : 0;
    if (wid == 0) mySum = warpReduceSum(mySum); //Final reduce within first warp
    if (tid_in_block == 0) {

        scalar_sum(curstep, mInd) = mySum;
        printf("mInd = %i, e = %.3f\n", mInd, scalar_sum(curstep, mInd));
    }
}

// Constructor
CudaMeasurementCalculations::CudaMeasurementCalculations() {
}
// Destructor
CudaMeasurementCalculations::~CudaMeasurementCalculations() {
    release();
}
// Initiator
bool CudaMeasurementCalculations::initiate(const Flag Flags, const SimulationParameters SimParam) {
    // Assert that we're not already initialized
    release();

    // Param

    mavrg_buff_size = SimParam.mavrg_buff_size;
    eavrg_buff_size = SimParam.eavrg_buff_size;
    mavrg_step = SimParam.mavrg_step;
    eavrg_step = SimParam.eavrg_step;
    bcount_mavrg = 0;
    bcount_eavrg = 0;
    do_mavrg = Flags.do_mavrg;
    do_eavrg = Flags.do_eavrg;
    N = SimParam.N;
    M = SimParam.M;
    spinTot = N * 3;
    tasksTot = M * N * 3;
    maxThreads = 512;
    maxBlocks = 1024; // must be less than maximum possible thread number
    numThreads = maxThreads;
    numBlocksX = std::min(((spinTot + numThreads - 1) / numThreads), maxBlocks);
    numBlocksY = M;
    blocks = { numBlocksX, numBlocksY, 1 };
    threads = { numThreads, 1, 1 };
    printf("numBlocks = %i\n", numBlocksX);



    // Allocate  class matrices
    if (do_mavrg) {
        mblock_gpu.Allocate(static_cast <long int>(3 * numBlocksX), static_cast <long int>(M));
        mbuff_gpu.Allocate(static_cast <long int>(3), static_cast <long int>(mavrg_buff_size), static_cast <long int>(M));

    }
    if (do_eavrg) {
        eblock_gpu.Allocate(static_cast <long int>(numBlocksX), static_cast <long int>(M));
        ebuff_gpu.Allocate(static_cast <long int>(eavrg_buff_size), static_cast <long int>(M));
    }


    // All initialized?
    if (cudaDeviceSynchronize() != cudaSuccess) {
        release();
        return false;
    }

    return true;
}
void CudaMeasurementCalculations::release() {
    if (do_mavrg) {
        mblock_gpu.Free();
        mbuff_gpu.Free();
    }
    if (do_eavrg) {
        eblock_gpu.Free();
        ebuff_gpu.Free();
    }
}
//Magnetization calculations
void CudaMeasurementCalculations::CalcMagnComponents(const CudaTensor<real, 3>& spin, int curstep) {
    mblock_gpu.zeros();
    GPUMagnSum << <blocks, threads >> > (spin, mblock_gpu, spinTot);
    GPUMagnFinalSum << <M, 1024 >> > (mblock_gpu, mbuff_gpu, numBlocksX, curstep);
    cudaDeviceSynchronize();
}
void CudaMeasurementCalculations::CalcEnergy(const CudaTensor<real, 3>& spin, const CudaTensor<real, 3>& eneff, int curstep) {
    eblock_gpu.zeros();
    GPUEnergySum << <blocks, threads >> > (spin, eneff, eblock_gpu, spinTot);
    GPUScalarFinalSum << <M, 1024 >> > (eblock_gpu, ebuff_gpu, numBlocksX, curstep);
    cudaDeviceSynchronize();
}
//Energy calculations


void CudaMeasurementCalculations::runMeasurement(const cudaLattice& gpuLattice, hostMeasurables& cpuMeasurables, const int curstep) {
    if (do_mavrg) {
        if (bcount_mavrg == mavrg_buff_size) {
            ////////////////////////////////////////////////////
            //////////CALL SAVING PRINTING HERE/////////////////
            ////////////////////////////////////////////////////

           // printf("buffering, curstep = %i, bcount = %i\n",curstep, bcount_avrg);
            //cpuMeasurables.mavg_buff.copy_sync(mbuff_gpu);
            cpuMeasurables.mavg_buff.zeros();
            mbuff_gpu.zeros();
            bcount_mavrg = 0;
        }

        if ((curstep % mavrg_step) == 0) {
            CalcMagnComponents(gpuLattice.emomM, bcount_mavrg);
            cudaDeviceSynchronize();
            bcount_mavrg++;
        }
    }
    if (do_eavrg) {
       // printf("buffering, curstep = %i, bcount = %i\n", curstep, bcount_avrg);
        if (bcount_eavrg == eavrg_buff_size) {
            ////////////////////////////////////////////////////
            //////////CALL SAVING PRINTING HERE/////////////////
            ////////////////////////////////////////////////////
            //cpuMeasurables.eavg_buff.copy_sync(ebuff_gpu);
            cpuMeasurables.eavg_buff.zeros();
            ebuff_gpu.zeros();
            bcount_eavrg = 0;
        }

        if ((curstep % eavrg_step) == 0) {
            CalcEnergy(gpuLattice.emomM, gpuLattice.eneff, bcount_eavrg);
            cudaDeviceSynchronize();
            bcount_eavrg++;
        }
    }

}






