#include "fortranData.hpp"

#include "real_type.h"

/////// Constants
char* FortranData::stt;
int* FortranData::SDEalgh;
//Sizes and steps
unsigned int* FortranData::rstep;
unsigned int* FortranData::nstep;
unsigned int* FortranData::Natom;
unsigned int* FortranData::Mensemble;
//Number of neighbours
unsigned int* FortranData::max_no_neigh;
unsigned int* FortranData::max_no_dmneigh;
unsigned int* FortranData::nHam;
//Parameters
real* FortranData::delta_t;
real* FortranData::gamma;
real* FortranData::k_bolt;
real* FortranData::mub;
real* FortranData::damping;
//For printing during the run
real* FortranData::binderc;
real* FortranData::mavg;

int* FortranData::mompar;
char* FortranData::initexc;
//Flags
unsigned int* FortranData::do_dm;
unsigned int* FortranData::do_jtensor;
unsigned int* FortranData::do_aniso;
//Measurables related constants
unsigned int* FortranData::do_cuda_meas;
unsigned int* FortranData::do_mavrg;
unsigned int* FortranData::do_eavrg;
unsigned int* FortranData::do_autocorr;
unsigned int* FortranData::mavrg_step;
unsigned int* FortranData::mavrg_buff_size;
unsigned int* FortranData::eavrg_step;
unsigned int* FortranData::eavrg_buff_size;

/////// Matrices
//Exchange
unsigned int * FortranData::aHam;
real* FortranData::ncoup;
unsigned int* FortranData::nlist;
unsigned int* FortranData::nlistsize;
//DMI
real* FortranData::dmvect;
unsigned int* FortranData::dmlist;
unsigned int* FortranData::dmlistsize;
//Anisotropy
real* FortranData::j_tensor;
real* FortranData::kaniso;
real* FortranData::eaniso;
unsigned int* FortranData::taniso;
real* FortranData::sb;
//Else
real* FortranData::external_field;
real* FortranData::btorque;
real* FortranData::temperature;
//Lattice quantities
real* FortranData::beff;
real* FortranData::b2eff;
real* FortranData::emomM;
real* FortranData::emom;
real* FortranData::emom2;
real* FortranData::mmom;
real* FortranData::mmom0;
real* FortranData::mmom2;
real* FortranData::mmomi;
//Measurables
real* FortranData::mavg_buff;
real* FortranData::mavg2_buff; 
real* FortranData::mavg4_buff;
real* FortranData::eavg_buff;
real* FortranData::eavg2_buff;

// GPU stuff
int* FortranData::gpu_mode;
int* FortranData::gpu_rng;
int* FortranData::gpu_rng_seed;

void FortranData::setFlagPointers(unsigned int* p_do_dm, unsigned int* p_do_jtensor, unsigned int* p_do_anisotropy, 
                              unsigned int* p_do_cuda_meas, unsigned int* p_do_mavrg, unsigned int* p_do_eavrg, unsigned int* p_do_autocorr) {
   do_dm = p_do_dm;
   do_jtensor = p_do_jtensor;
   do_aniso = p_do_anisotropy;
   do_cuda_meas = p_do_cuda_meas;
   do_mavrg = p_do_mavrg;
   do_eavrg = p_do_eavrg;
   do_autocorr = p_do_autocorr;
}

void FortranData::setConstantPointers(char* p_stt, int* p_SDEalgh, unsigned int* p_rstep, unsigned int* p_nstep, unsigned int* p_Natom, unsigned int* p_Mensemble, 
                                    unsigned int* p_max_no_neigh, unsigned int* p_nHam, unsigned int* p_max_no_dmneigh,  
                                    real* p_delta_t, real* p_gama, real* p_k_bolt, real* p_mub, real* p_mplambda1, 
                                    real* p_binderc, real* p_mavg,  int* p_mompar, char* p_initexc,    
                                    unsigned int* p_mavrg_step, unsigned int* p_mavrg_buff_size, 
                                    unsigned int* p_eavrg_step, unsigned int* p_eavrg_buff_size) {
   stt = p_stt;
   SDEalgh = p_SDEalgh;

   rstep = p_rstep;
   nstep = p_nstep;
   Natom = p_Natom;
   Mensemble = p_Mensemble;

   max_no_neigh = p_max_no_neigh;
   nHam = p_nHam;
   max_no_dmneigh = p_max_no_dmneigh;

   delta_t = p_delta_t;
   gamma = p_gama;
   k_bolt = p_k_bolt;
   mub = p_mub;
   damping = p_mplambda1;

   binderc = p_binderc;
   mavg = p_mavg;

   mompar = p_mompar;
   initexc = p_initexc;

   mavrg_step = p_mavrg_step;
   mavrg_buff_size = p_mavrg_buff_size; 
   eavrg_step = p_eavrg_step;
   eavrg_buff_size = p_eavrg_buff_size;
 
}

void FortranData::setHamiltonianPointers(real* p_ncoup, unsigned int* p_nlist,unsigned int* p_nlistsize, 
                                    real* p_dm_vect, unsigned int* p_dmlist,unsigned int* p_dmlistsize, 
                                    real* p_kaniso, real* p_eaniso, unsigned int* p_taniso, real* p_sb, 
                                    real* p_tens, unsigned int* p_aHam, 
                                    real* p_external_field, real* p_btorque, real* p_Temp_array){
   ncoup = p_ncoup;
   nlist = p_nlist;
   nlistsize = p_nlistsize;
   dmvect = p_dm_vect;
   dmlist = p_dmlist;
   dmlistsize = p_dmlistsize;
   kaniso = p_kaniso;
   eaniso = p_eaniso;
   taniso = p_taniso;
   sb = p_sb;
   j_tensor = p_tens;
   aHam = p_aHam;
   external_field = p_external_field;
   btorque = p_btorque;
   temperature = p_Temp_array;
}

void FortranData::setLatticePointers(real* p_beff, real* p_b2eff, real* p_emomM, real* p_emom, real* p_emom2, 
                                    real* p_mmom, real* p_mmom0, real* p_mmom2, real* p_mmomi) {
   beff = p_beff;
   b2eff = p_b2eff;
   emomM = p_emomM;
   emom = p_emom;
   emom2 = p_emom2;
   mmom = p_mmom;
   mmom0 = p_mmom0;
   mmom2 = p_mmom2;
   mmomi = p_mmomi;
}

void FortranData::setMeasurablePointers(real* p_mavg_buff, real* p_mavg2_buff, real* p_mavg4_buff,
                                    real* p_eavg_buff, real* p_eavg2_buff) {
   mavg_buff = p_mavg_buff;
   mavg2_buff = p_mavg2_buff; 
   mavg4_buff = p_mavg4_buff;
   eavg_buff = p_eavg_buff;
   eavg2_buff = p_eavg2_buff;
}

void FortranData::setInputDataPointers(int* p1, int* p2, int* p3) {
   gpu_mode = p1;
   gpu_rng = p2;
   gpu_rng_seed = p3;
}

// Fortran helpers
extern "C" void fortrandata_flags_(unsigned int* p_do_dm, unsigned int* p_do_jtensor, unsigned int* p_do_anisotropy, 
                                  unsigned int* p_do_cuda_avrg, unsigned int* p_do_autocorr) {
   FortranData::setFlagPointers(p_do_dm, p_do_jtensor, p_do_anisotropy, 
                                 p_do_cuda_avrg, p_do_autocorr);
}
extern "C" void fortrandata_constants_(char* p_stt, int* p_SDEalgh, unsigned int* p_rstep, unsigned int* p_nstep, unsigned int* p_Natom, unsigned int* p_Mensemble, 
                                    unsigned int* p_max_no_neigh, unsigned int* p_nHam, unsigned int* p_max_no_dmneigh,  
                                    real* p_delta_t, real* p_gama, real* p_k_bolt, real* p_mub, real* p_mplambda1, 
                                    real* p_binderc, real* p_mavg,  int* p_mompar, char* p_initexc,    
                                    unsigned int* p_avrg_step, unsigned int* p_avrg_buff, 
                                    unsigned int* p_eavrg_step, unsigned int* p_eavrg_buff_size) {
   FortranData::setConstantPointers(p_stt, p_SDEalgh, p_rstep, p_nstep, p_Natom, p_Mensemble, 
                                    p_max_no_neigh, p_nHam, p_max_no_dmneigh,  
                                    p_delta_t, p_gama, p_k_bolt, p_mub,  p_mplambda1, 
                                    p_binderc, p_mavg,  p_mompar, p_initexc,    
                                    p_avrg_step, p_avrg_buff, 
                                    p_eavrg_step, p_eavrg_buff_size);
}

extern "C" void fortrandata_sethamiltonian_(real* p_ncoup, unsigned int* p_nlist,unsigned int* p_nlistsize, 
                                    real* p_dm_vect, unsigned int* p_dmlist,unsigned int* p_dmlistsize, 
                                    real* p_kaniso, real* p_eaniso, unsigned int* p_taniso, real* p_sb, 
                                    real* p_tens, unsigned int* p_aHam, 
                                    real* p_external_field, real* p_btorque, real* p_Temp_array) {
   FortranData::setHamiltonianPointers(p_ncoup, p_nlist, p_nlistsize, 
                                    p_dm_vect, p_dmlist, p_dmlistsize, 
                                    p_kaniso, p_eaniso, p_taniso, p_sb, 
                                    p_tens, p_aHam, 
                                    p_external_field, p_btorque, p_Temp_array);
}

extern "C" void fortrandata_setlattice_(real* p_beff, real* p_b2eff, real* p_emomM, real* p_emom, real* p_emom2, 
                                    real* p_mmom, real* p_mmom0, real* p_mmom2, real* p_mmomi) {
   FortranData::setLatticePointers(p_beff, p_b2eff, p_emomM, p_emom, p_emom2, 
                                    p_mmom, p_mmom0, p_mmom2, p_mmomi);
}

extern "C" void fortrandata_setmeasurables_(real* p_mavg_buff, real* p_mavg2_buff, real* p_mavg4_buff,
                                    real* p_eavg_buff, real* p_eavg2_buff) {
   FortranData::setMeasurablePointers(p_mavg_buff, p_mavg2_buff, p_mavg4_buff,
                                    p_eavg_buff, p_eavg2_buff);
}

extern "C" void fortrandata_setinputdata_(int* p1, int* p2, int* p3) {
   FortranData::setInputDataPointers(p1, p2, p3);
}

