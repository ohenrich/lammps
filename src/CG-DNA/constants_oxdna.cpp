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
  real_flag = utils::strmatch(units.c_str(), "^real");
  if (real_flag) set_real_units();
}

// default to lj units
double ConstantsOxdna::d_cs = -0.4;
double ConstantsOxdna::d_cst = +0.34;
double ConstantsOxdna::d_chb = +0.4;
double ConstantsOxdna::d_cb = +0.4;
double ConstantsOxdna::d_cs_x = -0.34;
double ConstantsOxdna::d_cs_y = +0.3408;
double ConstantsOxdna::lambda_dh_one_prefactor = 0.3616455075438555; // = C1
double ConstantsOxdna::qeff_dh_pf_one_prefactor = 0.08173808693529228; // = C2

void ConstantsOxdna::set_real_units()
{
  // oxDNA 1 parameters in real units
  d_cs = -3.4072;
  d_cst = +2.89612;
  d_chb = d_cb = +3.4072;
  // oxDNA 2 parameters in real units
  d_cs_x = -2.89612;
  d_cs_y = +2.9029344;
  lambda_dh_one_prefactor = 0.05624154892; // = C1 * 8.518 * sqrt(k_B/4.142e-20)
  qeff_dh_pf_one_prefactor = 3.896883402; // = C2 * 5.597 * 8.518
};

}    // namespace LAMMPS_NS
