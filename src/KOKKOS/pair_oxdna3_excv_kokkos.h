/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(oxdna3/excv/kk,PairOxdna3ExcvKokkos<LMPDeviceType>);
PairStyle(oxdna3/excv/kk/device,PairOxdna3ExcvKokkos<LMPDeviceType>);
PairStyle(oxdna3/excv/kk/host,PairOxdna3ExcvKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_PAIR_OXDNA3_EXCV_KOKKOS_H
#define LMP_PAIR_OXDNA3_EXCV_KOKKOS_H

#include "pair_oxdna_excv_kokkos.h"
#include "pair_oxdna3_excv.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairOxdna3ExcvKokkos : public PairOxdnaExcvKokkos<DeviceType> {
 public:
  PairOxdna3ExcvKokkos(class LAMMPS *);
  ~PairOxdna3ExcvKokkos() {}
   void coeff(int, char **) override;
};

template<class DeviceType>
PairOxdna3ExcvKokkos<DeviceType>::PairOxdna3ExcvKokkos(LAMMPS *lmp) : PairOxdnaExcvKokkos<DeviceType>(lmp)
{
    this->oxdnaflag = PairOxdnaExcvKokkos<DeviceType>::EnabledOXDNAFlag::OXDNA2;
}

template<class DeviceType>
void PairOxdna3ExcvKokkos<DeviceType>::coeff(int narg, char **arg)
{
   PairOxdna3Excv::coeff_oxdna3_common(this, narg, arg);

   int ilo, ihi, jlo, jhi, nlo, nhi;
   utils::bounds(FLERR, arg[0], 1, this->atom->ntypes, ilo, ihi, this->error);
   utils::bounds(FLERR, arg[1], 1, this->atom->ntypes, jlo, jhi, this->error);

   assert((ilo == jlo) & (ihi == jhi));
   nlo = ilo;
   nhi = ihi;

   for (int i = 0; i <= nhi; i++) {
      for (int j = nlo; j <= nhi; j++) {
         for (int k = nlo; k <= nhi; k++) {
            for (int l = 0; l <= nhi; l++) {
               this->k_sigma4_bsbs.view_host()(i, j, k, l) = this->sigma4_bsbs[i][j][k][l];
               this->k_cut4_bsbs_ast.view_host()(i, j, k, l) = this->cut4_bsbs_ast[i][j][k][l];
               this->k_cut4sq_bsbs_ast.view_host()(i, j, k, l) = this->cut4sq_bsbs_ast[i][j][k][l];
               this->k_lj14_bsbs.view_host()(i, j, k, l) = this->lj14_bsbs[i][j][k][l];
               this->k_lj24_bsbs.view_host()(i, j, k, l) = this->lj24_bsbs[i][j][k][l];
               this->k_b4_bsbs.view_host()(i, j, k, l) = this->b4_bsbs[i][j][k][l];
               this->k_cut4_bsbs_c.view_host()(i, j, k, l) = this->cut4_bsbs_c[i][j][k][l];
               this->k_cut4sq_bsbs_c.view_host()(i, j, k, l) = this->cut4sq_bsbs_c[i][j][k][l];
            }
         }
      }
   }

   this->k_sigma4_bsbs.template modify<LMPHostType>();
   this->k_cut4_bsbs_ast.template modify<LMPHostType>();
   this->k_cut4sq_bsbs_ast.template modify<LMPHostType>();
   this->k_lj14_bsbs.template modify<LMPHostType>();
   this->k_lj24_bsbs.template modify<LMPHostType>();
   this->k_b4_bsbs.template modify<LMPHostType>();
   this->k_cut4_bsbs_c.template modify<LMPHostType>();
   this->k_cut4sq_bsbs_c.template modify<LMPHostType>();

   this->k_sigma4_bsbs.template sync<DeviceType>();
   this->k_cut4_bsbs_ast.template sync<DeviceType>();
   this->k_cut4sq_bsbs_ast.template sync<DeviceType>();
   this->k_lj14_bsbs.template sync<DeviceType>();
   this->k_lj24_bsbs.template sync<DeviceType>();
   this->k_b4_bsbs.template sync<DeviceType>();
   this->k_cut4_bsbs_c.template sync<DeviceType>();
   this->k_cut4sq_bsbs_c.template sync<DeviceType>();
}
}    // namespace LAMMPS_NS

#endif
#endif
