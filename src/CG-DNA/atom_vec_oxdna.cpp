/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "atom_vec_oxdna.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"

#include "atom_vec_ellipsoid.h"
#include "math_extra.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecOxdna::AtomVecOxdna(LAMMPS *lmp) : AtomVec(lmp)
{
  molecular = Atom::MOLECULAR;
  bonds_allow = 1;
  mass_type = PER_TYPE;
  forceclearflag = 1;

  atom->molecule_flag = 1;

  // strings with peratom variables to include in each AtomVec method
  // strings cannot contain fields in corresponding AtomVec default strings
  // order of fields in a string does not matter
  // except: fields_data_atom & fields_data_vel must match data file

  fields_grow = (char *) "id5p bb_pos";
  fields_copy = (char *) "id5p bb_pos";
  fields_comm = (char *) "bb_pos";
  fields_comm_vel = (char *) "";
  fields_reverse = (char *) "";
  fields_border = (char *) "id5p bb_pos";
  fields_border_vel = (char *) "";
  fields_exchange = (char *) "id5p bb_pos";
  fields_restart = (char *) "id5p bb_pos";
  fields_create = (char *) "";
  fields_data_atom = (char *) "id type x";
  fields_data_vel = (char *) "id v";

  setup_fields();

  if(!force->newton_bond) error->warning(FLERR,"Write_data command requires newton on to preserve 3'->5' bond polarity");
}

/* ---------------------------------------------------------------------- */

AtomVecOxdna::~AtomVecOxdna()
{
	
}

/* ----------------------------------------------------------------------
   set local copies of all grow ptrs used by this class, except defaults
   needed in replicate when 2 atom classes exist and it calls pack_restart()
------------------------------------------------------------------------- */

void AtomVecOxdna::grow_pointers()
{
  id5p = atom->id5p;
  bb_pos = atom->bb_pos;
}

/* ----------------------------------------------------------------------
    compute vector COM-sugar-phosphate backbone interaction site in oxDNA
------------------------------------------------------------------------- */

void AtomVecOxdna::compute_interaction_sites(double e1[3], double /*e2*/[3],
  double /*e3*/[3], double r[3])
{
  double d_cs=-0.4;

  r[0] = d_cs*e1[0];
  r[1] = d_cs*e1[1];
  r[2] = d_cs*e1[2];
}

/* ----------------------------------------------------------------------
   taking advantage of force_clear to calculate backbone positions
   each timestep
------------------------------------------------------------------------- */

void AtomVecOxdna::force_clear(int n, size_t nbytes)
{
  int a,b,i;

  double *qn,nx[3],ny[3],nz[3];
  
  // vectors COM-backbone site in lab frame
  double r_cs[3];

  double **x = atom->x;
  double **bb_pos = atom->bb_pos;
  
  int nlocal = atom->nlocal;
  
  AtomVecEllipsoid *avec = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
	
  // loop over nlocal atoms to set backbone positions

  for (i = 0; i < nlocal; i++) {
	
    qn=bonus[ellipsoid[i]].quat;
    MathExtra::q_to_exyz(qn,nx,ny,nz);
    compute_interaction_sites(nx,ny,nz,r_cs);
	
	bb_pos[i][0] = x[i][0] + r_cs[0];
	bb_pos[i][1] = x[i][1] + r_cs[1];
	bb_pos[i][2] = x[i][2] + r_cs[2];
  }
}

/* ----------------------------------------------------------------------
   initialize atom quantity 5' partner and backbone positions
------------------------------------------------------------------------- */

void AtomVecOxdna::data_atom_post(int ilocal)
{
  tagint *id5p = atom->id5p;
  id5p[ilocal] = -1;
  double **bb_pos = atom->bb_pos;
  bb_pos[0][ilocal] = -1;
  bb_pos[1][ilocal] = -1;
  bb_pos[2][ilocal] = -1;
}

/* ----------------------------------------------------------------------
   process bond information as per data file
   store 5' partner to inform 3'->5' bond directionality
------------------------------------------------------------------------- */

void AtomVecOxdna::data_bonds_post(int m, int num_bond, tagint atom1, tagint atom2,
                                   tagint id_offset)
{
  int n;
  tagint *id5p = atom->id5p;

  if (id_offset) {
    atom1 += id_offset;
    atom2 += id_offset;
  }

  if ((n = atom->map(atom1)) >= 0) { id5p[n] = atom2; }
}
