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
PairStyle(oxdna3/stk/kk,PairOxdna3StkKokkos<LMPDeviceType>);
PairStyle(oxdna3/stk/kk/device,PairOxdna3StkKokkos<LMPDeviceType>);
PairStyle(oxdna3/stk/kk/host,PairOxdna3StkKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_PAIR_OXDNA3_STK_KOKKOS_H
#define LMP_PAIR_OXDNA3_STK_KOKKOS_H

#include "pair_oxdna_stk_kokkos.h"
#include "pair_oxdna3_stk.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairOxdna3StkKokkos : public PairOxdnaStkKokkos<DeviceType> {
 public:
  PairOxdna3StkKokkos(class LAMMPS *);
  ~PairOxdna3StkKokkos() {}
   void coeff(int, char **) override;
};

template<class DeviceType>
PairOxdna3StkKokkos<DeviceType>::PairOxdna3StkKokkos(LAMMPS *lmp) : PairOxdnaStkKokkos<DeviceType>(lmp)
{
   PairOxdna3Stk::init_eta_st_oxdna3(this);
    this->oxdnaflag = PairOxdnaStkKokkos<DeviceType>::EnabledOXDNAFlag::OXDNA3;
}

template<class DeviceType>
void PairOxdna3StkKokkos<DeviceType>::coeff(int narg, char **arg)
{
   PairOxdna3Stk::coeff_oxdna3_common(this, narg, arg);

   this->coeff_set_tetramers_kokkos(narg, arg);
}

}    // namespace LAMMPS_NS

#endif
#endif
