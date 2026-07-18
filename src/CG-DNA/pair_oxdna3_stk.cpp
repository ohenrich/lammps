// clang-format off
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

#include "pair_oxdna3_stk.h"

#include "atom.h"
#include "comm.h"
#include "constants_oxdna.h"
#include "error.h"
#include "force.h"
#include "math_extra.h"
#include "memory.h"
#include "mf_oxdna.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "potential_file_reader.h"
#include "math_special.h"

#include <cmath>
#include <cstring>
#include <cassert>

using namespace LAMMPS_NS;
using namespace MathSpecial;
using namespace MFOxdna;

/* ---------------------------------------------------------------------- */

void PairOxdna3Stk::init_eta_st_oxdna3(PairOxdnaStk *oxdna_stk)
{
  oxdna_stk->eta_st[0][0] = 1.1217958408368172;
  oxdna_stk->eta_st[1][0] = 1.0712851690057155;
  oxdna_stk->eta_st[2][0] = 1.1161603311902566;
  oxdna_stk->eta_st[3][0] = 1.0052361315065244;

  oxdna_stk->eta_st[0][1] = 1.1217958408368172;
  oxdna_stk->eta_st[1][1] = 0.7892685731520542;
  oxdna_stk->eta_st[2][1] = 1.1022201982984874;
  oxdna_stk->eta_st[3][1] = 0.8658975520778347;

  oxdna_stk->eta_st[0][2] = 1.1217958408368172;
  oxdna_stk->eta_st[1][2] = 0.9896542231533637;
  oxdna_stk->eta_st[2][2] = 1.1088392608169480;
  oxdna_stk->eta_st[3][2] = 1.1217958408368172;

  oxdna_stk->eta_st[0][3] = 0.9300223683636719;
  oxdna_stk->eta_st[1][3] = 0.7694592613578328;
  oxdna_stk->eta_st[2][3] = 1.0007533199170144;
  oxdna_stk->eta_st[3][3] = 0.8593983791552220;
}

/* ---------------------------------------------------------------------- */

PairOxdna3Stk::PairOxdna3Stk(LAMMPS *lmp) : PairOxdnaStk(lmp)
{
  // sequence-specific stacking strength
  // A:0 C:1 G:2 T:3, 3'- [i][j] -5'
//
  // Moved 'eta_st' settings to static helper function since KOKKOS class of oxdna3/stk
  // inherits from PairOxdnaStk only, so cannot call this constructor to set
  // the eta_st values. Instead, we call this static function from the
  // constructor to set the eta_st values.
  //
  // KOKKOS base class goes through oxdna[1] in vanilla terms, due to GPU virtual function caveats.
  // But we make use of 'friend' classes so that we can access oxdna3 (vanilla) code in KOKKOS.
  //
  // These parameters settings are purely set-up routines and do not interact with any KOKKOS code.
  // Current known first point of actual use is within the coeff routine.
  init_eta_st_oxdna3(this);

  single_enable = 0;
  writedata = 0;
  trim_flag = 0;
}

/* ----------------------------------------------------------------------
  set coeffs - introduces new function to handle KOKKOS compatibility.
  Vanilla oxdna3 "coeff" literally just calls this "coeff_oxdna3_common"
  function. The structure here avoids messy inheritance issues in KOKKOS
  by not calling "PairOxdna3Stk::coeff" directly. We can also avoid
  code duplication of coeff within KOKKOS using this approach.

  "coeff_oxdna3_common" is static and takes a pointer to the base class
  PairOxdnaStk, which means it can be called from both the vanilla and
  KOKKOS versions.
  Can't use "coeff" directly since it is non-static - calling it would
  require an instance of the PairOxdna3Stk class, which is fine for vanilla
  but not for KOKKOS as we don't want KOKKOS to be a child class of PairOxdna3Stk.
------------------------------------------------------------------------- */

void PairOxdna3Stk::coeff_oxdna3_common(PairOxdnaStk *oxdna_stk, int narg, char **arg)
{
  init_eta_st_oxdna3(oxdna_stk);

  int count;

  if (narg != 4) oxdna_stk->error->all(FLERR,"Incorrect args for pair coefficients in oxdna3/stk, use potential file" + utils::errorurl(21));
  if (!oxdna_stk->allocated) oxdna_stk->allocate();

  int ilo,ihi,jlo,jhi,nlo,nhi;
  utils::bounds(FLERR,arg[0],1,oxdna_stk->atom->ntypes,ilo,ihi,oxdna_stk->error);
  utils::bounds(FLERR,arg[1],1,oxdna_stk->atom->ntypes,jlo,jhi,oxdna_stk->error);

  assert((ilo == jlo) & (ihi == jhi));
  nlo = ilo;
  nhi = ihi;

  if (nhi > 4) oxdna_stk->error->all(FLERR, "pair oxdna3/stk does not support more than 4 atom types for A, C, G and T");

  // stacking interaction
  count = 0;

  double tmp, theta_st4_0_one;
  double T, epsilon_st_one, xi_st_one, kappa_st_one, a_st_one, b_st_lo_one, b_st_hi_one;

  double a_st5_one, theta_st5_0_one, dtheta_st5_ast_one;
  double b_st5_one, dtheta_st5_c_one;

  double a_st6_one, theta_st6_0_one, dtheta_st6_ast_one;
  double b_st6_one, dtheta_st6_c_one;

  double a_st1_one, cosphi_st1_ast_one, b_st1_one, cosphi_st1_c_one;
  double a_st2_one, cosphi_st2_ast_one, b_st2_one, cosphi_st2_c_one;

  T = utils::numeric(FLERR,arg[2],false,oxdna_stk->lmp);

  for (int i = 0; i <= nhi; i++) { // type 0 for terminal j
    for (int j = 0; j <= nhi; j++) {
      for (int k = 0; k <= nhi; k++) {
        for (int l = 0; l <= nhi; l++) { // type 0 for terminal k
          oxdna_stk->cut_st_0[i][j][k][l] = 0.0;
          oxdna_stk->cut_st_c[i][j][k][l] = 0.0;
          oxdna_stk->cut_st_lo[i][j][k][l] = 0.0;
          oxdna_stk->cut_st_hi[i][j][k][l] = 0.0;
          oxdna_stk->a_st4[i][j][k][l] = 0.0;
          oxdna_stk->dtheta_st4_ast[i][j][k][l] = 0.0;
        }
      }
    }
  }

  if (oxdna_stk->comm->me == 0) {
    PotentialFileReader reader(oxdna_stk->lmp, arg[3], "oxdna3 potential", " (stk)");
    reader.set_bufsize(65336);
    char * line;
    std::string iloc, jloc, potential_name;

    while ((line = reader.next_line())) {
      try {
        ValueTokenizer values(line);
        iloc = values.next_string();
        jloc = values.next_string();
        potential_name = values.next_string();
        if (iloc == arg[0] && jloc == arg[1] && potential_name == "stk") {

          xi_st_one = values.next_double();
          kappa_st_one = values.next_double();
          epsilon_st_one = oxdna_stk->stacking_strength(xi_st_one, kappa_st_one, T);

          a_st_one = values.next_double();

          for (int i = nlo; i <= nhi; i++) {
            for (int j = nlo; j <= nhi; j++) {
              for (int k = nlo; k <= nhi; k++) {
                for (int l = nlo; l <= nhi; l++) {
                  oxdna_stk->cut_st_0[i][j][k][l] = values.next_double();
                  oxdna_stk->cut_st_0[i][j][k][0] += oxdna_stk->cut_st_0[i][j][k][l];
                  oxdna_stk->cut_st_0[0][j][k][l] += oxdna_stk->cut_st_0[i][j][k][l];
                  oxdna_stk->cut_st_0[0][j][k][0] += oxdna_stk->cut_st_0[i][j][k][l];
                }
              }
            }
          }
          for (int i = nlo; i <= nhi; i++) {
            for (int j = nlo; j <= nhi; j++) {
              for (int k = nlo; k <= nhi; k++) {
                for (int l = nlo; l <= nhi; l++) {
                  oxdna_stk->cut_st_c[i][j][k][l] = values.next_double();
                  oxdna_stk->cut_st_c[i][j][k][0] += oxdna_stk->cut_st_c[i][j][k][l];
                  oxdna_stk->cut_st_c[0][j][k][l] += oxdna_stk->cut_st_c[i][j][k][l];
                  oxdna_stk->cut_st_c[0][j][k][0] += oxdna_stk->cut_st_c[i][j][k][l];
                }
              }
            }
          }
          for (int i = nlo; i <= nhi; i++) {
            for (int j = nlo; j <= nhi; j++) {
              for (int k = nlo; k <= nhi; k++) {
                for (int l = nlo; l <= nhi; l++) {
                  oxdna_stk->cut_st_lo[i][j][k][l] = values.next_double();
                  oxdna_stk->cut_st_lo[i][j][k][0] += oxdna_stk->cut_st_lo[i][j][k][l];
                  oxdna_stk->cut_st_lo[0][j][k][l] += oxdna_stk->cut_st_lo[i][j][k][l];
                  oxdna_stk->cut_st_lo[0][j][k][0] += oxdna_stk->cut_st_lo[i][j][k][l];
                }
              }
            }
          }
          for (int i = nlo; i <= nhi; i++) {
            for (int j = nlo; j <= nhi; j++) {
              for (int k = nlo; k <= nhi; k++) {
                for (int l = nlo; l <= nhi; l++) {
                  oxdna_stk->cut_st_hi[i][j][k][l] = values.next_double();
                  oxdna_stk->cut_st_hi[i][j][k][0] += oxdna_stk->cut_st_hi[i][j][k][l];
                  oxdna_stk->cut_st_hi[0][j][k][l] += oxdna_stk->cut_st_hi[i][j][k][l];
                  oxdna_stk->cut_st_hi[0][j][k][0] += oxdna_stk->cut_st_hi[i][j][k][l];
                }
              }
            }
          }
          for (int i = nlo; i <= nhi; i++) {
            for (int j = nlo; j <= nhi; j++) {
              for (int k = nlo; k <= nhi; k++) {
                for (int l = nlo; l <= nhi; l++) {
                  oxdna_stk->a_st4[i][j][k][l] = values.next_double();
                  oxdna_stk->a_st4[i][j][k][0] += oxdna_stk->a_st4[i][j][k][l];
                  oxdna_stk->a_st4[0][j][k][l] += oxdna_stk->a_st4[i][j][k][l];
                  oxdna_stk->a_st4[0][j][k][0] += oxdna_stk->a_st4[i][j][k][l];
                }
              }
            }
          }

          theta_st4_0_one = values.next_double();

          for (int i = nlo; i <= nhi; i++) {
            for (int j = nlo; j <= nhi; j++) {
              for (int k = nlo; k <= nhi; k++) {
                for (int l = nlo; l <= nhi; l++) {
                  oxdna_stk->dtheta_st4_ast[i][j][k][l] = values.next_double();
                  oxdna_stk->dtheta_st4_ast[i][j][k][0] += oxdna_stk->dtheta_st4_ast[i][j][k][l];
                  oxdna_stk->dtheta_st4_ast[0][j][k][l] += oxdna_stk->dtheta_st4_ast[i][j][k][l];
                  oxdna_stk->dtheta_st4_ast[0][j][k][0] += oxdna_stk->dtheta_st4_ast[i][j][k][l];
                }
              }
            }
          }

          a_st5_one = values.next_double();
          theta_st5_0_one = values.next_double();
          dtheta_st5_ast_one = values.next_double();
          a_st6_one = values.next_double();
          theta_st6_0_one = values.next_double();
          dtheta_st6_ast_one = values.next_double();
          a_st1_one = values.next_double();
          cosphi_st1_ast_one = values.next_double();
          a_st2_one = values.next_double();
          cosphi_st2_ast_one = values.next_double();

          break;
        } else continue;
      } catch (std::exception &e) {
        oxdna_stk->error->one(FLERR, "Problem parsing oxDNA3 potential file: {}", e.what());
      }
    }
    if ((iloc != arg[0]) || (jloc != arg[1]) || (potential_name != "stk"))
      oxdna_stk->error->one(FLERR, "No corresponding stk potential found in file {} for pair type {} {}",
                 arg[3], arg[0], arg[1]);



    // calculate sequence-averaged parameters for terminal base step j-k
    for (int i = nlo; i <= nhi; i++) {
      for (int j = nlo; j <= nhi; j++) {
        for (int k = nlo; k <= nhi; k++) {
          oxdna_stk->cut_st_0[i][j][k][0] /= nhi;
          oxdna_stk->cut_st_c[i][j][k][0] /= nhi;
          oxdna_stk->cut_st_lo[i][j][k][0] /= nhi;
          oxdna_stk->cut_st_hi[i][j][k][0] /= nhi;
          oxdna_stk->a_st4[i][j][k][0] /= nhi;
          oxdna_stk->dtheta_st4_ast[i][j][k][0] /= nhi;
        }
      }
    }
    for (int j = nlo; j <= nhi; j++) {
      for (int k = nlo; k <= nhi; k++) {
        for (int l = nlo; l <= nhi; l++) {
          oxdna_stk->cut_st_0[0][j][k][l] /= nhi;
          oxdna_stk->cut_st_c[0][j][k][l] /= nhi;
          oxdna_stk->cut_st_lo[0][j][k][l] /= nhi;
          oxdna_stk->cut_st_hi[0][j][k][l] /= nhi;
          oxdna_stk->a_st4[0][j][k][l] /= nhi;
          oxdna_stk->dtheta_st4_ast[0][j][k][l] /= nhi;
        }
      }
    }
    for (int j = nlo; j <= nhi; j++) {
      for (int k = nlo; k <= nhi; k++) {
        oxdna_stk->cut_st_0[0][j][k][0] /= powint(nhi,2);
        oxdna_stk->cut_st_c[0][j][k][0] /= powint(nhi,2);
        oxdna_stk->cut_st_lo[0][j][k][0] /= powint(nhi,2);
        oxdna_stk->cut_st_hi[0][j][k][0] /= powint(nhi,2);
        oxdna_stk->a_st4[0][j][k][0] /= powint(nhi,2);
        oxdna_stk->dtheta_st4_ast[0][j][k][0] /= powint(nhi,2);
      }
    }

  }

  MPI_Bcast(&epsilon_st_one, 1, MPI_DOUBLE, 0, oxdna_stk->world);
  MPI_Bcast(&a_st_one, 1, MPI_DOUBLE, 0, oxdna_stk->world);

  MPI_Bcast(&oxdna_stk->cut_st_0[0][0][0][0], 625, MPI_DOUBLE, 0, oxdna_stk->world);
  MPI_Bcast(&oxdna_stk->cut_st_c[0][0][0][0], 625, MPI_DOUBLE, 0, oxdna_stk->world);
  MPI_Bcast(&oxdna_stk->cut_st_lo[0][0][0][0], 625, MPI_DOUBLE, 0, oxdna_stk->world);
  MPI_Bcast(&oxdna_stk->cut_st_hi[0][0][0][0], 625, MPI_DOUBLE, 0, oxdna_stk->world);
  MPI_Bcast(&oxdna_stk->a_st4[0][0][0][0], 625, MPI_DOUBLE, 0, oxdna_stk->world);

  MPI_Bcast(&theta_st4_0_one, 1, MPI_DOUBLE, 0, oxdna_stk->world);

  MPI_Bcast(&oxdna_stk->dtheta_st4_ast[0][0][0][0], 625, MPI_DOUBLE, 0, oxdna_stk->world);

  MPI_Bcast(&a_st5_one, 1, MPI_DOUBLE, 0, oxdna_stk->world);
  MPI_Bcast(&theta_st5_0_one, 1, MPI_DOUBLE, 0, oxdna_stk->world);
  MPI_Bcast(&dtheta_st5_ast_one, 1, MPI_DOUBLE, 0, oxdna_stk->world);
  MPI_Bcast(&a_st6_one, 1, MPI_DOUBLE, 0, oxdna_stk->world);
  MPI_Bcast(&theta_st6_0_one, 1, MPI_DOUBLE, 0, oxdna_stk->world);
  MPI_Bcast(&dtheta_st6_ast_one, 1, MPI_DOUBLE, 0, oxdna_stk->world);
  MPI_Bcast(&a_st1_one, 1, MPI_DOUBLE, 0, oxdna_stk->world);
  MPI_Bcast(&cosphi_st1_ast_one, 1, MPI_DOUBLE, 0, oxdna_stk->world);
  MPI_Bcast(&a_st2_one, 1, MPI_DOUBLE, 0, oxdna_stk->world);
  MPI_Bcast(&cosphi_st2_ast_one, 1, MPI_DOUBLE, 0, oxdna_stk->world);

  // smoothing - determined through continuity and differentiability

  // smoothing strength coincidentally identical for all pairs ij, hence use AAAA tetramer value below
    b_st_lo_one = 2*a_st_one*exp(-a_st_one*(oxdna_stk->cut_st_lo[1][1][1][1]-oxdna_stk->cut_st_0[1][1][1][1]))*
      2*a_st_one*exp(-a_st_one*(oxdna_stk->cut_st_lo[1][1][1][1]-oxdna_stk->cut_st_0[1][1][1][1]))*
      (1-exp(-a_st_one*(oxdna_stk->cut_st_lo[1][1][1][1]-oxdna_stk->cut_st_0[1][1][1][1])))*
      (1-exp(-a_st_one*(oxdna_stk->cut_st_lo[1][1][1][1]-oxdna_stk->cut_st_0[1][1][1][1])))
      /(4*((1-exp(-a_st_one*(oxdna_stk->cut_st_lo[1][1][1][1] -oxdna_stk->cut_st_0[1][1][1][1])))*
      (1-exp(-a_st_one*(oxdna_stk->cut_st_lo[1][1][1][1]-oxdna_stk->cut_st_0[1][1][1][1])))-
      (1-exp(-a_st_one*(oxdna_stk->cut_st_c[1][1][1][1] -oxdna_stk->cut_st_0[1][1][1][1])))*
      (1-exp(-a_st_one*(oxdna_stk->cut_st_c[1][1][1][1]-oxdna_stk->cut_st_0[1][1][1][1])))));

  // smoothing strength coincidentally identical for all pairs ij, hence use AAAA tetramer value below
    b_st_hi_one = 2*a_st_one*exp(-a_st_one*(oxdna_stk->cut_st_hi[1][1][1][1]-oxdna_stk->cut_st_0[1][1][1][1]))*
      2*a_st_one*exp(-a_st_one*(oxdna_stk->cut_st_hi[1][1][1][1]-oxdna_stk->cut_st_0[1][1][1][1]))*
      (1-exp(-a_st_one*(oxdna_stk->cut_st_hi[1][1][1][1]-oxdna_stk->cut_st_0[1][1][1][1])))*
      (1-exp(-a_st_one*(oxdna_stk->cut_st_hi[1][1][1][1]-oxdna_stk->cut_st_0[1][1][1][1])))
      /(4*((1-exp(-a_st_one*(oxdna_stk->cut_st_hi[1][1][1][1] -oxdna_stk->cut_st_0[1][1][1][1])))*
      (1-exp(-a_st_one*(oxdna_stk->cut_st_hi[1][1][1][1]-oxdna_stk->cut_st_0[1][1][1][1])))-
      (1-exp(-a_st_one*(oxdna_stk->cut_st_c[1][1][1][1] -oxdna_stk->cut_st_0[1][1][1][1])))*
      (1-exp(-a_st_one*(oxdna_stk->cut_st_c[1][1][1][1]-oxdna_stk->cut_st_0[1][1][1][1])))));

  b_st5_one = a_st5_one*a_st5_one*dtheta_st5_ast_one*dtheta_st5_ast_one/
      (1-a_st5_one*dtheta_st5_ast_one*dtheta_st5_ast_one);
  dtheta_st5_c_one = 1/(a_st5_one*dtheta_st5_ast_one);

  b_st6_one = a_st6_one*a_st6_one*dtheta_st6_ast_one*dtheta_st6_ast_one/
      (1-a_st6_one*dtheta_st6_ast_one*dtheta_st6_ast_one);
  dtheta_st6_c_one = 1/(a_st6_one*dtheta_st6_ast_one);

  b_st1_one = a_st1_one*a_st1_one*cosphi_st1_ast_one*cosphi_st1_ast_one/
      (1-a_st1_one*cosphi_st1_ast_one*cosphi_st1_ast_one);
  cosphi_st1_c_one = 1/(a_st1_one*cosphi_st1_ast_one);

  b_st2_one = a_st2_one*a_st2_one*cosphi_st2_ast_one*cosphi_st2_ast_one/
      (1-a_st2_one*cosphi_st2_ast_one*cosphi_st2_ast_one);
  cosphi_st2_c_one = 1/(a_st2_one*cosphi_st2_ast_one);


  // parameters, uniform or depending on base step
  for (int i = nlo; i <= nhi; i++) {
    for (int j = nlo; j <= nhi; j++) {

      oxdna_stk->epsilon_st[i][j] = epsilon_st_one * oxdna_stk->eta_st[i-1][j-1];

      oxdna_stk->a_st[i][j] = a_st_one;
      oxdna_stk->b_st_lo[i][j] = b_st_lo_one;
      oxdna_stk->b_st_hi[i][j] = b_st_hi_one;
      oxdna_stk->theta_st4_0[i][j] = theta_st4_0_one;

      oxdna_stk->a_st5[i][j] = a_st5_one;
      oxdna_stk->theta_st5_0[i][j] = theta_st5_0_one;
      oxdna_stk->dtheta_st5_ast[i][j] = dtheta_st5_ast_one;
      oxdna_stk->b_st5[i][j] = b_st5_one;
      oxdna_stk->dtheta_st5_c[i][j] = dtheta_st5_c_one;

      oxdna_stk->a_st6[i][j] = a_st6_one;
      oxdna_stk->theta_st6_0[i][j] = theta_st6_0_one;
      oxdna_stk->dtheta_st6_ast[i][j] = dtheta_st6_ast_one;
      oxdna_stk->b_st6[i][j] = b_st6_one;
      oxdna_stk->dtheta_st6_c[i][j] = dtheta_st6_c_one;

      oxdna_stk->a_st1[i][j] = a_st1_one;
      oxdna_stk->cosphi_st1_ast[i][j] = cosphi_st1_ast_one;
      oxdna_stk->b_st1[i][j] = b_st1_one;
      oxdna_stk->cosphi_st1_c[i][j] = cosphi_st1_c_one;

      oxdna_stk->a_st2[i][j] = a_st2_one;
      oxdna_stk->cosphi_st2_ast[i][j] = cosphi_st2_ast_one;
      oxdna_stk->b_st2[i][j] = b_st2_one;
      oxdna_stk->cosphi_st2_c[i][j] = cosphi_st2_c_one;

    }
  }

  // parameters depending on tetramer
  for (int i = 0; i <= nhi; i++) { // type 0 for terminal j
    for (int j = nlo; j <= nhi; j++) {
      for (int k = nlo; k <= nhi; k++) {
        for (int l = 0; l <= nhi; l++) { // type 0 for terminal k

              oxdna_stk->cut_st_lc[i][j][k][l] = oxdna_stk->cut_st_lo[i][j][k][l]
                - a_st_one*exp(-a_st_one*(oxdna_stk->cut_st_lo[i][j][k][l]-oxdna_stk->cut_st_0[i][j][k][l]))*
                (1-exp(-a_st_one*(oxdna_stk->cut_st_lo[i][j][k][l]-oxdna_stk->cut_st_0[i][j][k][l])))/b_st_lo_one;

              oxdna_stk->cut_st_hc[i][j][k][l] = oxdna_stk->cut_st_hi[i][j][k][l]
                - a_st_one*exp(-a_st_one*(oxdna_stk->cut_st_hi[i][j][k][l]-oxdna_stk->cut_st_0[i][j][k][l]))*
                (1-exp(-a_st_one*(oxdna_stk->cut_st_hi[i][j][k][l]-oxdna_stk->cut_st_0[i][j][k][l])))/b_st_hi_one;

          oxdna_stk->cutsq_st_hc[i][j][k][l] = oxdna_stk->cut_st_hc[i][j][k][l]*oxdna_stk->cut_st_hc[i][j][k][l];

          tmp = 1 - exp(-(oxdna_stk->cut_st_c[i][j][k][l]-oxdna_stk->cut_st_0[i][j][k][l]) * a_st_one);
          oxdna_stk->shift_st[i][j][k][l] = epsilon_st_one * oxdna_stk->eta_st[j-1][k-1] * tmp * tmp;

              oxdna_stk->b_st4[i][j][k][l] = oxdna_stk->a_st4[i][j][k][l]*oxdna_stk->a_st4[i][j][k][l]*oxdna_stk->dtheta_st4_ast[i][j][k][l]*
                oxdna_stk->dtheta_st4_ast[i][j][k][l]/(1-oxdna_stk->a_st4[i][j][k][l]*oxdna_stk->dtheta_st4_ast[i][j][k][l]*oxdna_stk->dtheta_st4_ast[i][j][k][l]);
              oxdna_stk->dtheta_st4_c[i][j][k][l] = 1/(oxdna_stk->a_st4[i][j][k][l]*oxdna_stk->dtheta_st4_ast[i][j][k][l]);

        }
      }
      oxdna_stk->setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) oxdna_stk->error->all(FLERR,"Incorrect args for pair coefficients in oxdna3/stk" + utils::errorurl(21));

}

void PairOxdna3Stk::coeff(int narg, char **arg) { coeff_oxdna3_common(this, narg, arg); }
