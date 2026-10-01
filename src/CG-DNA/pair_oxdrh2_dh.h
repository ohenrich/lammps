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
PairStyle(oxdrh2/dh,PairOxdrh2Dh);
// clang-format on
#else

#ifndef LMP_PAIR_OXDRH2_DH_H
#define LMP_PAIR_OXDRH2_DH_H

#include "pair_oxdna2_dh.h"
#include "nucleotide_oxdna.h"

namespace LAMMPS_NS {

class PairOxdrh2Dh : public PairOxdna2Dh {
 public:
  PairOxdrh2Dh(class LAMMPS *lmp) : PairOxdna2Dh(lmp) {}
  inline void compute_backbone_site(int type, double e1[3], double e2[3],
    double e3[3], double rbk[3]) const override
  {
    NucleotideOxdna2 oxdna2;
    NucleotideOxrna2 oxrna2;
    switch (type) {
      case 0:
        oxrna2.backbone_site<0>(e1, nullptr, e3, rbk);
        break;
      case 1:
      case 2:
      case 3:
      case 4:
        oxdna2.backbone_site<0>(e1, e2, nullptr, rbk);
        break;
      case 5:
      case 6:
      case 7:
        oxrna2.backbone_site<0>(e1, nullptr, e3, rbk);
        break;
    }
  };
};

}    // namespace LAMMPS_NS

#endif
#endif
