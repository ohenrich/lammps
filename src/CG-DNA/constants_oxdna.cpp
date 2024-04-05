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

#include "constants_oxdna.h"

namespace LAMMPS_NS {

ConstantsOxdna::ConstantsOxdna(class LAMMPS *lmp) : Pointers(lmp)
{
  // set oxDNA units
  units = update->unit_style;
  lj_flag = (strcmp(units.c_str(), "lj") == 0);
  set_oxdna_units();
}

double ConstantsOxdna::d_cs = 0;
double ConstantsOxdna::d_cst = 0;
double ConstantsOxdna::d_chb = 0;
double ConstantsOxdna::d_cb = 0;

void ConstantsOxdna::set_oxdna_units()
{
  if (lj_flag) {
    // oxDNA 1 parameters in lj units
    d_cs = -0.4;
    d_cst = +0.34;
    d_chb = d_cb = +0.4;
  } else {
    // oxDNA 1 parameters in real units
    d_cs = -3.4072;
    d_cst = +2.89612;
    d_chb = d_cb = +3.4072;
  }
};

}    // namespace LAMMPS_NS
