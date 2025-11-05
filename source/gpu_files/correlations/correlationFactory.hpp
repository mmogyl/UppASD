#pragma once

#include "c_headers.hpp"
#include "tensor.hpp"
#include "real_type.h"
#include "fortranData.hpp"
#include "gpuStructures.hpp"
#include "correlation.hpp"

#include "fortranCorrelation.hpp"

#if defined(HIP_V)
#include "gpuCorrelations.hpp"
#elif defined(CUDA_V)
#include "gpuCorrelations.cuh"
#endif

#include <iostream>

class CorrelationFactory
{
public:
    // could be moved to a .cu file, but the function was so short, so I implemented it
    // directly in the header
    static std::unique_ptr<Correlation> create(const deviceLattice& gpuLattice, hostLattice& cpuLattice, const Flag Flags, const SimulationParameters SimParam, const hostCorrelations& cpuCorrelations)
    {

        char do_gpu_correlations = 'N';
        //printf("\n 1 - do_gpu_correlations = %c\n", do_gpu_correlations);

        if (*FortranData::do_sc != 'N')  do_gpu_correlations = *FortranData::do_gpu_correlations;
        //printf("\n 2 - do_gpu_correlations = %c\n", do_gpu_correlations);

        if (do_gpu_correlations == 'Y')
        {
            std::cout << "GpuCorrelation used" << std::endl;
            return std::make_unique<GpuCorrelations>(Flags, SimParam, gpuLattice, cpuCorrelations);
        }
        else
        {
            std::cout << "FortranCorrelation used" << std::endl;
            return std::make_unique<FortranCorrelation>(
                gpuLattice.emomM,
                gpuLattice.emom,
                gpuLattice.mmom,
                cpuLattice.emomM,
                cpuLattice.emom,
                cpuLattice.mmom
            );
        }
    }
};
