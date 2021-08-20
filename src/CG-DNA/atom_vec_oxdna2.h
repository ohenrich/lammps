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
AtomStyle(oxdna2,AtomVecOxdna2);
// clang-format on
#else

#ifndef LMP_ATOM_VEC_OXDNA2_H
#define LMP_ATOM_VEC_OXDNA2_H

#include "atom_vec_oxdna.h"

namespace LAMMPS_NS {

class AtomVecOxdna2 : public AtomVecOxdna {
 public:
  AtomVecOxdna2(class LAMMPS *);
  virtual ~AtomVecOxdna2();
  virtual void compute_interaction_sites(double *, double *, double *, double *);
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

*/
