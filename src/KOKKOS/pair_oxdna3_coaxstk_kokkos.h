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
PairStyle(oxdna3/coaxstk/kk,PairOxdna3CoaxstkKokkos<LMPDeviceType>);
PairStyle(oxdna3/coaxstk/kk/device,PairOxdna3CoaxstkKokkos<LMPDeviceType>);
PairStyle(oxdna3/coaxstk/kk/host,PairOxdna3CoaxstkKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_PAIR_OXDNA3_COAXSTK_KOKKOS_H
#define LMP_PAIR_OXDNA3_COAXSTK_KOKKOS_H

#include "pair_oxdna2_coaxstk_kokkos.h"
#include "pair_oxdna3_coaxstk.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairOxdna3CoaxstkKokkos : public PairOxdna2CoaxstkKokkos<DeviceType> {
 public:
  PairOxdna3CoaxstkKokkos(class LAMMPS *);
  ~PairOxdna3CoaxstkKokkos() {}
  void coeff(int, char **) override;
};

template<class DeviceType>
PairOxdna3CoaxstkKokkos<DeviceType>::PairOxdna3CoaxstkKokkos(LAMMPS *lmp) : PairOxdna2CoaxstkKokkos<DeviceType>(lmp)
{
  this->oxdnaflag = PairOxdna2CoaxstkKokkos<DeviceType>::EnabledOXDNAFlag::OXDNA3;
  this->init_eta_cxst_oxdna3();
}

template<class DeviceType>
void PairOxdna3CoaxstkKokkos<DeviceType>::coeff(int narg, char **arg)
{
  this->coeff_oxdna3_common(narg, arg);
}

}    // namespace LAMMPS_NS

#endif
#endif
