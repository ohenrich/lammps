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
PairStyle(oxrna2/dh,PairOxrna2Dh);
// clang-format on
#else

#ifndef LMP_PAIR_OXRNA2_DH_H
#define LMP_PAIR_OXRNA2_DH_H

#include "nucleotide_oxdna.h"
#include "pair_oxdna2_dh.h"

namespace LAMMPS_NS {

class PairOxrna2Dh : public PairOxdna2Dh {
 public:
  PairOxrna2Dh(class LAMMPS *lmp) : PairOxdna2Dh(lmp) {}
  // inline below has to be here in the header file, otherwise KOKKOS
  // compilation fails due to undefined vtable symbols.
  void compute_backbone_site(double e1[3], double /*e2*/[3], double e3[3],
                             double rbk[3]) const override
  {
    NucleotideOxrna2 oxrna2;
    oxrna2.backbone_site(e1, nullptr, e3, rbk);
  };
};

}    // namespace LAMMPS_NS

#endif
#endif
