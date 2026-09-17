/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------
   Contributing author: Oliver Henrich (University of Strathclyde, Glasgow)
------------------------------------------------------------------------- */

#include "pair_oxdrh2_hbond.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairOxdrh2Hbond::PairOxdrh2Hbond(LAMMPS *lmp) : PairOxdnaHbond(lmp)
{
  single_enable = 0;
  writedata = 0;
  trim_flag = 0;

  // sequence-specific base-pairing strength
  // DNA A:0 C:1 G:2 T:3, 3'- [i][j] -5'
  // RNA A:4 C:5 G:6 U:7, 3'- [i][j] -5'

 for (int i=0; i<8; i++) {
    for (int j=0; j<8; j++) {
      alpha_hb[i][j] = 1.00000;
    }
  }

  alpha_hb[0][7] = 0.8066666666666666;
  alpha_hb[7][0] = 0.8066666666666666;

  alpha_hb[1][6] = 1.18;
  alpha_hb[6][1] = 1.18;

  alpha_hb[2][5] = 1.0733333333333335;
  alpha_hb[5][2] = 1.0733333333333335;

  alpha_hb[3][4] = 0.9133333333333334;
  alpha_hb[4][3] = 0.9133333333333334;
}
