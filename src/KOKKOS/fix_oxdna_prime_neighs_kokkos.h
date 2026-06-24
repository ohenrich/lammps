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

#ifdef FIX_CLASS
// clang-format off
FixStyle(oxdna/prime/neighs/kk,FixOxdnaPrimeNeighsKokkos<LMPDeviceType>);
FixStyle(oxdna/prime/neighs/kk/device,FixOxdnaPrimeNeighsKokkos<LMPDeviceType>);
FixStyle(oxdna/prime/neighs/kk/host,FixOxdnaPrimeNeighsKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_FIX_OXDNA_PRIME_NEIGHS_KOKKOS_H
#define LMP_FIX_OXDNA_PRIME_NEIGHS_KOKKOS_H

#include "fix.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

struct TagFixOxdnaPrimeNeighsPrecomputeBondPrimeNeighs {};

template<class DeviceType>
class FixOxdnaPrimeNeighsKokkos : public Fix {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixOxdnaPrimeNeighsKokkos(class LAMMPS *, int, char **);
  ~FixOxdnaPrimeNeighsKokkos() override;

  int setmask() override;
  void min_setup_pre_force(int);
  void min_pre_force(int) override;
  void setup_pre_force(int) override;
  void pre_force(int) override;

  // 0-3 : atom a, atom b, id3p[a], id5p[b] for each bond.
  typename AT::t_int_1d_4 d_bond_prime_neighs;
  int nbondlist;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixOxdnaPrimeNeighsPrecomputeBondPrimeNeighs, const int &) const;

 private:
  class NeighborKokkos *neighborKK;

  typename AT::t_int_2d_lr bondlist;
  typename AT::t_tagint_1d tag;
  typename AT::t_tagint_1d id5p;
  typename AT::t_tagint_1d id3p;

  int map_style;
  DAT::tdual_int_1d k_map_array;
  dual_hash_type k_map_hash;

  bigint last_precompute_lastcall;

  void compute_prime_neighs();
};

}    // namespace LAMMPS_NS
#endif
#endif