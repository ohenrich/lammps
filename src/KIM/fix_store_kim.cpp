/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Axel Kohlmeyer (Temple U),
                         Ryan S. Elliott (UMN)
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   This program is free software; you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by the Free
   Software Foundation; either version 2 of the License, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
   more details.

   You should have received a copy of the GNU General Public License along with
   this program; if not, see <https://www.gnu.org/licenses>.

   Linking LAMMPS statically or dynamically with other modules is making a
   combined work based on LAMMPS. Thus, the terms and conditions of the GNU
   General Public License cover the whole combination.

   In addition, as a special exception, the copyright holders of LAMMPS give
   you permission to combine LAMMPS with free software programs or libraries
   that are released under the GNU LGPL and with code included in the standard
   release of the "kim-api" under the CDDL (or modified versions of such code,
   with unchanged license). You may copy and distribute such a system following
   the terms of the GNU GPL for LAMMPS and the licenses of the other code
   concerned, provided that you include the source code of that other code
   when and as the GNU GPL requires distribution of source code.

   Note that people who make modified versions of LAMMPS are not obligated to
   grant this special exception for their modified versions; it is their choice
   whether to do so. The GNU General Public License gives permission to release
   a modified version without this exception; this exception also makes it
   possible to release a modified version which carries forward this exception.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Designed for use with the kim-api-2.0.2 (and newer) package
------------------------------------------------------------------------- */

#include "fix_store_kim.h"
#include <cstring>
extern "C" {
#include "KIM_SimulatorModel.h"
}
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixStoreKIM::FixStoreKIM(LAMMPS *lmp, int narg, char **arg)
  : Fix(lmp, narg, arg), simulator_model(NULL), model_name(NULL),
    model_units(NULL), user_units(NULL)
{
  if (narg != 3) error->all(FLERR,"Illegal fix STORE/KIM command");
}

/* ---------------------------------------------------------------------- */

FixStoreKIM::~FixStoreKIM()
{
  // free associated storage

  if (simulator_model) {
    KIM_SimulatorModel *sm = (KIM_SimulatorModel *)simulator_model;
    KIM_SimulatorModel_Destroy(&sm);
    simulator_model = NULL;
  }

  if (model_name) {
    char *mn = (char *)model_name;
    delete[] mn;
    model_name = NULL;
  }

  if (model_units) {
    char *mu = (char *)model_units;
    delete[] mu;
    model_units = NULL;
  }
  if (user_units) {
    char *uu = (char *)user_units;
    delete[] uu;
    user_units = NULL;
  }
}

/* ---------------------------------------------------------------------- */

int FixStoreKIM::setmask()
{
  int mask = 0;
  return mask;
}


/* ---------------------------------------------------------------------- */

void FixStoreKIM::setptr(const char *name, void *ptr)
{
  if (strcmp(name,"simulator_model") == 0) {
    if (simulator_model) {
      KIM_SimulatorModel *sm = (KIM_SimulatorModel *)simulator_model;
      KIM_SimulatorModel_Destroy(&sm);
    }
    simulator_model = ptr;
  } else if (strcmp(name,"model_name") == 0) {
    if (model_name) {
      char *mn = (char *)model_name;
      delete[] mn;
    }
    model_name = ptr;
  } else if (strcmp(name,"model_units") == 0) {
    if (model_units) {
      char *mu = (char *)model_units;
      delete[] mu;
    }
    model_units = ptr;
  } else if (strcmp(name,"user_units") == 0) {
    if (user_units) {
      char *uu = (char *)user_units;
      delete[] uu;
    }
    user_units = ptr;
  } else error->all(FLERR,"Unknown property in fix STORE/KIM");
}

/* ---------------------------------------------------------------------- */

void *FixStoreKIM::getptr(const char *name)
{
  if (strcmp(name,"simulator_model") == 0) return simulator_model;
  else if (strcmp(name,"model_name") == 0) return model_name;
  else if (strcmp(name,"model_units") == 0) return model_units;
  else if (strcmp(name,"user_units") == 0) return user_units;
  else return NULL;
}