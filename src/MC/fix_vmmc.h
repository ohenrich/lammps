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
FixStyle(vmmc,FixVMMC);
// clang-format on
#else

#ifndef LMP_FIX_VMMC_H
#define LMP_FIX_VMMC_H

#include "fix.h"
#include "VMMC.h"

namespace LAMMPS_NS {

class FixVMMC : public Fix {
 public:
  FixVMMC(class LAMMPS *, int, char **);
  ~FixVMMC() override;
  int setmask() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void pre_exchange() override;
  double memory_usage() override;
  void write_restart(FILE *) override;
  void restart(char *) override;
  void *extract(const char *, int &) override;
  vmmc::VMMC *vmmc; // libVMMC

 private:
  int exclusion_group, exclusion_group_bit;
  int nvmmc_type, nevery, seed;
  int ncycles, nvmmcmoves;
  double patomtrans, pmctot;
  int exchmode;            // exchange ATOM, MOLECULE not supported
  int movemode;            // move ATOM, MOLECULE not supported
  class Region *region;    // vmmc region
  char *idregion;          // vmmc region id
  bool pressure_flag;      // true if user specified reservoir pressure

  int groupbitall;            // group bitmask for inserted atoms
  int ngroups;                // number of group-ids for inserted atoms
  char **groupstrings;        // list of group-ids for inserted atoms
  int ngrouptypes;            // number of type-based group-ids for inserted atoms
  char **grouptypestrings;    // list of type-based group-ids for inserted atoms
  int *grouptypebits;         // list of type-based group bitmasks
  int *grouptypes;            // list of type-based group types

  int mc_active;              // 1 during MC trials, otherwise 0

  int vmmc_nmax;
  int max_region_attempts;
  double temperature;
  double max_translate, max_rotate;
  double beta;
  double volume, pressure;
  double xlo, xhi, ylo, yhi, zlo, zhi;
  double region_xlo, region_xhi, region_ylo, region_yhi, region_zlo, region_zhi;
  double region_volume;
  double **cutsq;
  imageint imagezero;
  double overlap_cutoffsq;    // square distance cutoff for overlap
  int overlap_flag;

  class Pair *pair;
  class RanPark *random_equal;
  class RanPark *random_unequal;
  class NeighList *list;

  void options(int, char **);

  // VMMC routines
  double pair_energy(unsigned int, const double*, const double*,
                          unsigned int, const double*, const double*);
  double pair_potential(int, int, int, int, double, double, double, double &);
  unsigned int interactions(unsigned int, const double*, const double*, unsigned int*);
  void post_move(unsigned int, const double*, const double*);

};

}    // namespace LAMMPS_NS


#endif
#endif
