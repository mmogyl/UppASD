#pragma once

#include "c_headers.hpp"
#include "real_type.h"

// NAME         TYPE     DIMENSION   DESCRIPTION
//
// mompar       int       1          Parametrization of magnetic moment magnitudes (0=no)
// initexc      char      1          Mode of excitation of initial magnetic moments (I=vacancies, R=two magnon
// Raman, F=no)
//
// emom         real     (3,N,M)     Current unit moment vector
// emom2        real     (3,N,M)     Final (or temporary) unit moment vector
// emomM        real     (3,N,M)     Current magnetic moment vector
// mmom         real     (N,M)       Magnitude of magnetic moments
// mmom0        real     (N,M)       Starting magnitude of magnetic moments
// mmom2        real     (N,M)       Temporary value of magnitude of magnetic moments
// mmomi        real     (N,M)       Inverse of magnitude of magnetic moments
//
// mrod         real     (3,N,M)     Rotated magnetic moments
// btherm       real     (3,N,M)     Thermal stochastic field
// bloc         real     (3,N,M)     Local effective field
// bdup         real     (3,N,M)     Resulting effective field
//
// beff         real     (3,N,M)     Total effective field from application of Hamiltonian
// b2eff        real     (3,N,M)     Temporary storage of magnetic field
// btorque      real     (3,N,M)     Spin transfer torque
// emom         real     (3,N,M)     Current unit moment vector
// emom2        real     (3,N,M)     Final (or temporary) unit moment vector
// emomM        real     (3,N,M)     Current magnetic moment vector
// mmom         real     (N,M)       Magnitude of magnetic moments
// delta_t      real      1          Time step
// temperature  real     (N)         Temperature
//
// stt          char      1          Method to handle spin transfer torque
// sb 			real 	 (N)	     Ratio between cubic and uniaxial anisotropy

class FortranData {
public:
   // Scalars
   static char* stt;
   static int* SDEalgh;

   static unsigned int* rstep;
   static unsigned int* nstep;
   static unsigned int* Natom;
   static unsigned int* Mensemble;
   
   static unsigned int* max_no_neigh;
   static unsigned int* nHam;
   static unsigned int* max_no_dmneigh;

   static real* delta_t;
   static real* gamma;
   static real* k_bolt;
   static real* mub;
   static real* damping;

   static real* binderc;
   static real* mavg;

   static int* mompar;
   static char* initexc;

   static unsigned int* do_dm;

   static unsigned int*
       do_jtensor;  // Information on weather the exchange coupling tensor should be used or not
   static unsigned int* do_aniso;  // Information on weather the anisotropy should be used or not
   
   static unsigned int* do_cuda_meas;
   static unsigned int* do_mavrg;
   static unsigned int* do_eavrg;
   static unsigned int* do_autocorr;
   static unsigned int* mavrg_step;
   static unsigned int* mavrg_buff_size;
   static unsigned int* eavrg_step;
   static unsigned int* eavrg_buff_size;
   // Matrices / vectors
   static unsigned int * aHam;

   static real* ncoup;
   static unsigned int* nlist;
   static unsigned int* nlistsize;

   static real* dmvect;
   static unsigned int* dmlist;
   static unsigned int* dmlistsize;

   static real* j_tensor;

   static real* kaniso;
   static real* eaniso;
   static unsigned int* taniso;
   static real* sb;

   static real* beff;
   static real* b2eff;
   static real* emomM;
   static real* emom;
   static real* emom2;
   static real* external_field;
   static real* mmom;
   static real* btorque;
   static real* temperature;
   static real* mmom0;
   static real* mmom2;
   static real* mmomi;

   // Input
   static int* gpu_mode;
   static int* gpu_rng;
   static int* gpu_rng_seed;

    static real* mavg_buff;
    static real* mavg2_buff; 
    static real* mavg4_buff;
    static real* eavg_buff;
    static real* eavg2_buff;

   // Initiators
   static void setFlagPointers(unsigned int* p_do_dm, unsigned int* p_do_jtensor, unsigned int* p_do_anisotropy, 
                              unsigned int* p_do_cuda_meas, unsigned int* p_do_mavrg, unsigned int* p_do_eavrg, unsigned int* p_do_autocorr);


   static void setConstantPointers(char* p_stt, int* p_SDEalgh, unsigned int* p_rstep, unsigned int* p_nstep, unsigned int* p_Natom, unsigned int* p_Mensemble, 
                                    unsigned int* p_max_no_neigh, unsigned int* p_nHam, unsigned int* p_max_no_dmneigh,  
                                    real* p_delta_t, real* p_gama, real* p_k_bolt, real* p_mub, real* p_mplambda1, 
                                    real* p_binderc, real* p_mavg,  int* p_mompar, char* p_initexc,    
                                    unsigned int* p_avrg_step, unsigned int* p_avrg_buff, 
                                    unsigned int* p_eavrg_step, unsigned int* p_eavrg_buff_size);

   static void setHamiltonianPointers(real* p_ncoup, unsigned int* p_nlist,unsigned int* p_nlistsize, 
                                    real* p_dm_vect, unsigned int* p_dmlist,unsigned int* p_dmlistsize, 
                                    real* p_kaniso, real* p_eaniso, unsigned int* p_taniso, real* p_sb, 
                                    real* p_tens, unsigned int* p_aHam, 
                                    real* p_external_field, real* p_btorque, real* p_Temp_array);

    static void setLatticePointers(real* p_beff, real* p_b2eff, real* p_emomM, real* p_emom, real* p_emom2, real* p_mmom, real* p_mmom0, real* p_mmom2, real* p_mmomi);

    static void setMeasurablePointers(real* p_mavg_buff, real* p_mavg2_buff, real* p_mavg4_buff,
                                    real* p_eavg_buff, real* p_eavg2_buff);

   static void setInputDataPointers(int* p1, int* p2, int* p3);
};

