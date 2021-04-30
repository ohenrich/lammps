/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Axel Kohlmeyer (Temple U)
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(meam/spline/omp,PairMEAMSplineOMP);
// clang-format on
#else

#ifndef LMP_PAIR_MEAM_SPLINE_OMP_H
#define LMP_PAIR_MEAM_SPLINE_OMP_H

#include "pair_meam_spline.h"
#include "thr_omp.h"

namespace LAMMPS_NS {

class PairMEAMSplineOMP : public PairMEAMSpline, public ThrOMP {

 public:
  PairMEAMSplineOMP(class LAMMPS *);

  virtual void compute(int, int);
  virtual double memory_usage();

 private:
  template <int EVFLAG, int EFLAG>
  void eval(int iifrom, int iito, ThrData * const thr);
};

}

#endif
#endif
