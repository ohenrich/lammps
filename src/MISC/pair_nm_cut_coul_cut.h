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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(nm/cut/coul/cut,PairNMCutCoulCut);
// clang-format on
#else

#ifndef LMP_PAIR_NM_CUT_COUL_CUT_H
#define LMP_PAIR_NM_CUT_COUL_CUT_H

#include "pair.h"

namespace LAMMPS_NS {

class PairNMCutCoulCut : public Pair {
 public:
  PairNMCutCoulCut(class LAMMPS *);
  virtual ~PairNMCutCoulCut();

  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  void write_data(FILE *);
  void write_data_all(FILE *);
  double single(int, int, int, int, double, double, double, double &);
  void *extract(const char *, int &);

 protected:
  double cut_lj_global,cut_coul_global;
  double **cut_lj,**cut_ljsq;
  double **cut_coul,**cut_coulsq;
  double **e0,**r0,**nn, **mm;
  double **nm,**e0nm,**r0n,**r0m,**offset;

  void allocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair style nm/cut/coul/cut requires atom attribute q

The atom style defined does not have this attribute.

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

*/
