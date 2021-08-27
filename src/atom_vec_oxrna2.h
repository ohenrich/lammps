/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef ATOM_CLASS
// clang-format off
AtomStyle(oxrna2,AtomVecOxrna2);
// clang-format on
#else

#ifndef LMP_ATOM_VEC_OXRNA2_H
#define LMP_ATOM_VEC_OXRNA2_H

#include "atom_vec_oxdna.h"

namespace LAMMPS_NS {

class AtomVecOxrna2 : public AtomVecOxdna {
 public:
  AtomVecOxrna2(class LAMMPS *);
  ~AtomVecOxrna2();
  virtual void compute_interaction_sites(double *, double *, double *, double *);

 private:
  tagint *id5p;
  double **bb_pos;
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

*/
