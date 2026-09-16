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

#include "pair_oxrna2_hbond.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairOxrna2Hbond::PairOxrna2Hbond(LAMMPS *lmp) : PairOxdnaHbond(lmp)
{
  single_enable = 0;
  writedata = 0;
  trim_flag = 0;

  // sequence-specific base-pairing strength
  // A:0/4 C:1/5 G:2/6 U:3/7, 5'- [i][j] -3'

 for (int i=0; i<8; i++) {
    for (int j=0; j<8; j++) {
      alpha_hb[i][j] = 1.00000;
    }
  }

  alpha_hb[0][3] = 0.94253;
  alpha_hb[0][7] = 0.94253;
  alpha_hb[4][3] = 0.94253;
  alpha_hb[4][7] = 0.94253;

  alpha_hb[1][2] = 1.22288;
  alpha_hb[1][6] = 1.22288;
  alpha_hb[5][2] = 1.22288;
  alpha_hb[5][6] = 1.22288;

  alpha_hb[2][1] = 1.22288;
  alpha_hb[2][5] = 1.22288;
  alpha_hb[6][1] = 1.22288;
  alpha_hb[6][5] = 1.22288;

  alpha_hb[2][3] = 0.58655;
  alpha_hb[2][7] = 0.58655;
  alpha_hb[6][3] = 0.58655;
  alpha_hb[6][7] = 0.58655;

  alpha_hb[3][0] = 0.94253;
  alpha_hb[3][4] = 0.94253;
  alpha_hb[7][0] = 0.94253;
  alpha_hb[7][4] = 0.94253;

  alpha_hb[3][2] = 0.58655;
  alpha_hb[3][6] = 0.58655;
  alpha_hb[7][2] = 0.58655;
  alpha_hb[7][6] = 0.58655;

}
