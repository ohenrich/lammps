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

#ifndef CONSTANTS_OXDNA_H
#define CONSTANTS_OXDNA_H

#include "update.h"

namespace LAMMPS_NS {

class ConstantsOxdna : protected Pointers {
 public:
  ConstantsOxdna(class LAMMPS *lmp);
  virtual ~ConstantsOxdna(){};

  // oxDNA 1 getters
  static double get_d_cs() { return d_cs; }
  static double get_d_cst() { return d_cst; }
  static double get_d_chb() { return d_chb; }
  static double get_d_cb() { return d_cb; }

 private:
  std::string units;
  bool lj_flag;
  void set_oxdna_units();

  // oxDNA 1 parameters
  static double d_cs, d_cst, d_chb, d_cb;
};

}    // namespace LAMMPS_NS

#endif
