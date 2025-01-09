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
FixStyle(langevin/wall,FixLangevinWall);
// clang-format on
#else

#ifndef LMP_FIX_LANGEVIN_WALL_H
#define LMP_FIX_LANGEVIN_WALL_H

#include "fix_langevin.h"
#include "fix_wall.h"

namespace LAMMPS_NS {

class FixLangevinWall : public FixLangevin {
 public:
  FixLangevinWall(class LAMMPS *, int, char **);
  ~FixLangevinWall() override;
  void post_force(int) override;
  void init();

 protected:
  template <int Tp_TSTYLEATOM, int Tp_GJF, int Tp_TALLY, int Tp_BIAS, int Tp_RMASS, int Tp_ZERO>
  void post_force_templated();
  void angmom_thermostat();
  FixWall *wallfix[2];
  double *rad;

};
}    // namespace LAMMPS_NS
#endif
#endif
