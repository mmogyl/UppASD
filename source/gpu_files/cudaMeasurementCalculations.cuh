#pragma once

#include <curand.h>
#include "c_headers.hpp"
#include "tensor.cuh"
#include "real_type.h"
#include "cudaStructures.hpp"
#include <numeric>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <complex>

class CudaMeasurementCalculations {
private:

    unsigned int numThreads;
    unsigned int numBlocksX;
    unsigned int numBlocksY;
    std::size_t mavrg_buff_size; // buffer size for m, m_proj, m^2 proj,mavg_buff_projch
    std::size_t eavrg_buff_size; // buffer size for m, m_proj, m^2 proj,mavg_buff_projch
    std::size_t bcount_mavrg;
    std::size_t bcount_eavrg;
    std::size_t mavrg_step;
    std::size_t eavrg_step;

    std::size_t cc_step;
    std::size_t N;
    std::size_t M;
    unsigned int  spinTot;
    unsigned int  tasksTot;
    unsigned int maxThreads;
    unsigned int maxBlocks;
    dim3 blocks;
    dim3 threads;

    real r_mid;
    std::size_t nq;
    real nainv;
    real qfac;
    std::complex<real> iqfac;

    /////////bcount for cumu?
    bool do_mavrg;
    bool do_eavrg;
    bool do_cc;

    //Block variables
    CudaTensor<real, 2> mblock_gpu; 
    CudaTensor<real, 3> mbuff_gpu; 
    CudaTensor<real, 2> eblock_gpu;
    CudaTensor<real, 2> ebuff_gpu;
    

    // Buffer variables 
    Tensor<real, 3> mavg_buff;     // 3 x buff x M
    Tensor<real, 2> eavg_buff;     // buff x M


    // Variable calculations
    void CalcMagnComponents(const CudaTensor<real, 3>& spin, int curstep); //mmom 3 x N x M to mavg_buff 3 x M x buff
    void CalcEnergy(const CudaTensor<real, 3>& spin, const CudaTensor<real, 3>& eneff, int curstep); //mmom 3 x N x M times beff 3 x N x M to 
    // energy per spin  N x M

public:
    // Constructor
    CudaMeasurementCalculations();
    // Destructor
    ~CudaMeasurementCalculations();

    // Initiator
    bool initiate(const Flag Flags, const SimulationParameters SimParam);
    // Releaser
    void release();
    // Measurements
    void runMeasurement(const cudaLattice& gpuLattice, hostMeasurables& cpuMeasurables, const int curstep);
};

