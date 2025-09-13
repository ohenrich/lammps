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
   Contributing authors:
    Lester Hedges (Bristol, UK)
    Ford Cadman, Oliver Henrich (University of Strathclyde, Glasgow)
------------------------------------------------------------------------- */
//#define DIPOLE

#include "fix_vmmc.h"
#include "VMMC.h"
#include "MersenneTwister.h"
#include "pair_lj_cut_dipole_cut.h"

#include "atom.h"
#include "atom_vec.h"
#include "bond.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "group.h"
#include "math_const.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "pair.h"
#include "random_park.h"
#include "region.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <exception>
#include <cassert>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;
using namespace vmmc;

// large energy value used to signal overlap

static constexpr double MAXENERGYSIGNAL = 1.0e100;

// this must be lower than MAXENERGYSIGNAL
// by a large amount, so that it is still
// less than total energy when negative
// energy contributions are added to MAXENERGYSIGNAL

static constexpr double MAXENERGYTEST = 1.0e50;

enum { NONE, MOVEATOM };    // movemode

/* ---------------------------------------------------------------------- */

FixVMMC::FixVMMC(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), region(nullptr), idregion(nullptr), groupstrings(nullptr),
    grouptypestrings(nullptr), grouptypebits(nullptr), grouptypes(nullptr),
    random_equal(nullptr), random_unequal(nullptr), list(nullptr)
{
  if (narg < 10) utils::missing_cmd_args(FLERR, "fix vmmc", error);

  if (atom->molecular == Atom::TEMPLATE)
    error->all(FLERR,"Fix vmmc does not (yet) work with atom_style template");

  dynamic_group_allow = 1;

  vector_flag = 1;
  size_vector = 8;
  global_freq = 1;
  extvector = 0;
  restart_global = 1;
  time_depend = 1;

  ngroups = 0;
  ngrouptypes = 0;

  // required args

  nevery = utils::inumeric(FLERR, arg[3], false, lmp);
  nvmmcmoves = utils::inumeric(FLERR, arg[4], false, lmp);
  nvmmc_type = utils::expand_type_int(FLERR, arg[5], Atom::ATOM, lmp);
  seed = utils::inumeric(FLERR, arg[6], false, lmp);
  temperature = utils::numeric(FLERR, arg[7], false, lmp);
  max_translate = utils::numeric(FLERR, arg[8], false, lmp);
  max_rotate = utils::numeric(FLERR, arg[9], false, lmp);

  if (nevery <= 0) error->all(FLERR, "Illegal fix vmmc command");
  if (nvmmcmoves < 0) error->all(FLERR, "Illegal fix vmmc command");
  if (seed <= 0) error->all(FLERR, "Illegal fix vmmc command");
  if (temperature < 0.0)
    error->all(FLERR, "Illegal fix vmmc command");
  if (max_translate < 0.0) error->all(FLERR, "Illegal fix vmmc command");

  // read options from end of input line

  options(narg-10,&arg[10]);

  // random number generator, same for all procs

  random_equal = new RanPark(lmp,seed);

  // random number generator, not the same for all procs

  random_unequal = new RanPark(lmp,seed);

  // error check on lower box boundaries

  if (domain->boxlo[0] != 0.0 || domain->boxlo[1] != 0.0 || domain->boxlo[2] != 0.0) {
    error->all(FLERR,"Fix vmmc requires box boundaries xlo = ylo = zlo = 0.0");
  }

  // error checks on region and its extent being inside simulation box

  region_xlo = region_xhi = region_ylo = region_yhi = region_zlo = region_zhi = 0.0;
  if (region) {
    if (region->bboxflag == 0)
      error->all(FLERR,"Fix vmmc region does not support a bounding box");
    if (region->dynamic_check())
      error->all(FLERR,"Fix vmmc region cannot be dynamic");

    region_xlo = region->extent_xlo;
    region_xhi = region->extent_xhi;
    region_ylo = region->extent_ylo;
    region_yhi = region->extent_yhi;
    region_zlo = region->extent_zlo;
    region_zhi = region->extent_zhi;

    // estimate region volume using MC trials

    double coord[3];
    int inside = 0;
    int attempts = 10000000;
    for (int i = 0; i < attempts; i++) {
      coord[0] = region_xlo + random_equal->uniform() * (region_xhi-region_xlo);
      coord[1] = region_ylo + random_equal->uniform() * (region_yhi-region_ylo);
      coord[2] = region_zlo + random_equal->uniform() * (region_zhi-region_zlo);
      if (region->match(coord[0],coord[1],coord[2]) != 0)
        inside++;
    }

    double max_region_volume = (region_xhi - region_xlo) *
      (region_yhi - region_ylo) * (region_zhi - region_zlo);

    region_volume = max_region_volume * static_cast<double>(inside) / static_cast<double>(attempts);
  }

  // compute the number of MC cycles that occur nevery timesteps

  ncycles = nvmmcmoves;

  // set up reneighboring

  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;

  mc_active = 0;

  vmmc_nmax = 0;
  vmmc = nullptr;

}

/* ----------------------------------------------------------------------
   parse optional parameters at end of input line
------------------------------------------------------------------------- */

void FixVMMC::options(int narg, char **arg)
{
  if (narg < 0) error->all(FLERR,"Illegal fix vmmc command");

  // defaults

  movemode = NONE;
  patomtrans = 0.0;
  pmctot = 0.0;
  region_volume = 0;
  max_region_attempts = 1000;
  exclusion_group = 0;
  exclusion_group_bit = 0;
  pressure_flag = false;
  pressure = 0.0;
  ngroups = 0;
  int ngroupsmax = 0;
  groupstrings = nullptr;
  ngrouptypes = 0;
  int ngrouptypesmax = 0;
  grouptypestrings = nullptr;
  grouptypes = nullptr;
  grouptypebits = nullptr;
  overlap_cutoffsq = 0.0;
  overlap_flag = 0;

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"mcmoves") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix vmmc command");
      patomtrans = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (patomtrans < 0)
        error->all(FLERR,"Illegal fix vmmc command");
      pmctot = patomtrans;
      if (pmctot <= 0)
        error->all(FLERR,"Illegal fix vmmc command");
      iarg += 4;
    } else if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix vmmc command");
      region = domain->get_region_by_id(arg[iarg+1]);
      if (!region) error->all(FLERR,"Region {} for fix vmmc does not exist",arg[iarg+1]);
      idregion = utils::strdup(arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"pressure") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix vmmc command");
      pressure = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      pressure_flag = true;
      iarg += 2;
    } else if (strcmp(arg[iarg],"group") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix vmmc command");
      if (ngroups >= ngroupsmax) {
        ngroupsmax = ngroups+1;
        groupstrings = (char **)
          memory->srealloc(groupstrings,
                           ngroupsmax*sizeof(char *),
                           "fix_vmmc:groupstrings");
      }
      groupstrings[ngroups] = utils::strdup(arg[iarg+1]);
      ngroups++;
      iarg += 2;
    } else if (strcmp(arg[iarg],"grouptype") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix vmmc command");
      if (ngrouptypes >= ngrouptypesmax) {
        ngrouptypesmax = ngrouptypes+1;
        grouptypes = (int*) memory->srealloc(grouptypes,ngrouptypesmax*sizeof(int),
                         "fix_vmmc:grouptypes");
        grouptypestrings = (char**)
          memory->srealloc(grouptypestrings,
                           ngrouptypesmax*sizeof(char *),
                           "fix_vmmc:grouptypestrings");
      }
      grouptypes[ngrouptypes] = utils::expand_type_int(FLERR, arg[iarg+1], Atom::ATOM, lmp);
      grouptypestrings[ngrouptypes] = utils::strdup(arg[iarg+2]);
      ngrouptypes++;
      iarg += 3;
    } else if (strcmp(arg[iarg],"overlap_cutoff") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix vmmc command");
      double rtmp = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      overlap_cutoffsq = rtmp*rtmp;
      overlap_flag = 1;
      iarg += 2;
    } else error->all(FLERR,"Illegal fix vmmc command");
  }
}

/* ---------------------------------------------------------------------- */

FixVMMC::~FixVMMC()
{
  delete[] idregion;
  delete random_equal;
  delete random_unequal;

  if (ngroups > 0) {
    for (int igroup = 0; igroup < ngroups; igroup++)
      delete[] groupstrings[igroup];
    memory->sfree(groupstrings);
  }

  if (ngrouptypes > 0) {
    memory->destroy(grouptypes);
    memory->destroy(grouptypebits);
    for (int igroup = 0; igroup < ngrouptypes; igroup++)
      delete[] grouptypestrings[igroup];
    memory->sfree(grouptypestrings);
  }

  // delete exclusion group created in init()
  // delete molecule group created in init()
  // unset neighbor exclusion settings made in init()
  // not necessary if group and neighbor classes already destroyed
  //   when LAMMPS exits

  if (exclusion_group_bit && group) {
    auto group_id = std::string("FixVMMC:vmmc_exclusion_group:") + id;
    try {
      group->assign(group_id + " delete");
    } catch (std::exception &e) {
      if (comm->me == 0)
        fprintf(stderr, "Error deleting group %s: %s\n", group_id.c_str(), e.what());
    }
  }

  delete vmmc;
}

/* ---------------------------------------------------------------------- */

int FixVMMC::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixVMMC::init()
{
  if (!atom->mass) error->all(FLERR, "Fix vmmc requires per atom type masses");
  if (atom->rmass_flag && (comm->me == 0))
    error->warning(FLERR, "Fix vmmc will use per atom type masses for velocity initialization");

  // set index and check validity of region

  if (idregion) {
    region = domain->get_region_by_id(idregion);
    if (!region) error->all(FLERR, "Region {} for fix vmmc does not exist", idregion);
  }

  if (region) {
    if (region->bboxflag == 0)
      error->all(FLERR,"Fix vmmc region does not support a bounding box");
    if (region->dynamic_check())
      error->all(FLERR,"Fix vmmc region cannot be dynamic");

    region_xlo = region->extent_xlo;
    region_xhi = region->extent_xhi;
    region_ylo = region->extent_ylo;
    region_yhi = region->extent_yhi;
    region_zlo = region->extent_zlo;
    region_zhi = region->extent_zhi;

    if ((region_xlo < domain->boxlo[0]) || (region_xhi > domain->boxhi[0]) ||
        (region_ylo < domain->boxlo[1]) || (region_yhi > domain->boxhi[1]) ||
        (region_zlo < domain->boxlo[2]) || (region_zhi > domain->boxhi[2]))
      error->all(FLERR,"Fix vmmc region extends outside simulation box");
   }

  // set probabilities for MC moves

  if (nvmmcmoves > 0) {
    movemode = MOVEATOM;
    if (pmctot == 0.0) {
        patomtrans = 1.0;
    }
    else {
      patomtrans /= pmctot;
    }
  } else movemode = NONE;

  if (domain->dimension == 2)
    error->all(FLERR,"Cannot use fix vmmc in a 2d simulation");

  // check that no deletable atoms are in atom->firstgroup
  // deleting such an atom would not leave firstgroup atoms first

  if (atom->firstgroup >= 0) {
    int *mask = atom->mask;
    int firstgroupbit = group->bitmask[atom->firstgroup];

    int flag = 0;
    for (int i = 0; i < atom->nlocal; i++)
      if ((mask[i] == groupbit) && (mask[i] && firstgroupbit)) flag = 1;

    int flagall;
    MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);

    if (flagall)
      error->all(FLERR,"Cannot do VMMC on atoms in atom_modify first group");
  }

  // compute beta
  beta = 1.0/(force->boltz*temperature);

  imagezero = ((imageint) IMGMAX << IMG2BITS) |
             ((imageint) IMGMAX << IMGBITS) | IMGMAX;

  // warning if group id is "all"

  if ((comm->me == 0) && (groupbit & 1))
    error->warning(FLERR, "Fix vmmc is being applied "
                   "to the default group all");

  // construct group bitmask for all new atoms
  // aggregated over all group keywords

  groupbitall = 1 | groupbit;
  for (int igroup = 0; igroup < ngroups; igroup++) {
    int jgroup = group->find(groupstrings[igroup]);
    if (jgroup == -1)
      error->all(FLERR,"Could not find specified fix vmmc group ID");
    groupbitall |= group->bitmask[jgroup];
  }

  // construct group type bitmasks
  // not aggregated over all group keywords

  if (ngrouptypes > 0) {
    memory->create(grouptypebits,ngrouptypes,"fix_vmmc:grouptypebits");
    for (int igroup = 0; igroup < ngrouptypes; igroup++) {
      int jgroup = group->find(grouptypestrings[igroup]);
      if (jgroup == -1)
        error->all(FLERR,"Could not find specified fix vmmc group ID");
      grouptypebits[igroup] = group->bitmask[jgroup];
    }
  }

  // initialise global to local index mapping
  if(atom->map_style != Atom::MAP_YES) atom->map_init();

  // need a full neighbor list, built every Nevery steps
  neighbor->add_request(this, NeighConst::REQ_FULL);

  // initialise the libVMMC callback functions
  using namespace std::placeholders;
  vmmc::CallbackFunctions callbacks;

  callbacks.pairEnergyCallback = std::bind(&FixVMMC::pair_energy, this, _1, _2, _3, _4, _5, _6);
  callbacks.interactionsCallback = std::bind(&FixVMMC::interactions, this, _1, _2, _3, _4);
  callbacks.postMoveCallback = std::bind(&FixVMMC::post_move, this, _1, _2, _3);

  double coordinates[domain->dimension*atom->natoms];
  double orientations[domain->dimension*atom->natoms];
  bool isIsotropic[atom->natoms];

  // set dummy particle coordinates and orientations for libVMMC object initialization
  // coordinates and orientations are set in pre_exchange() prior to VMMC move 
  for (int ii=0; ii<atom->natoms; ii++) {

    coordinates[domain->dimension*ii + 0] = 0.0;
    coordinates[domain->dimension*ii + 1] = 0.0;
    coordinates[domain->dimension*ii + 2] = 0.0;

    orientations[domain->dimension*ii + 0] = 1.0;
    orientations[domain->dimension*ii + 1] = 0.0;
    orientations[domain->dimension*ii + 2] = 0.0;

    isIsotropic[ii] = false; // set flag to enable orientational moves

  }

  unsigned int maxInteractions= 1000;
  double boxSize[3];
  boxSize[0] = domain->boxhi[0];
  boxSize[1] = domain->boxhi[1];
  boxSize[2] = domain->boxhi[2];

  // initialize libVMMC object and RNG seed
  vmmc = new VMMC(atom->natoms, domain->dimension, coordinates, orientations,
      max_translate, max_rotate, 0.5, 0.5, maxInteractions, &boxSize[0], isIsotropic, true, callbacks);

  vmmc->rng.setSeed(seed);
  printf("\nVMMC seed %d\n",  vmmc->rng.getSeed());

  // set beta in libVMMC
  vmmc->setBeta(beta);

}

/* ----------------------------------------------------------------------
   neighbor callback to inform pair style of neighbor list to use regular
------------------------------------------------------------------------- */

void FixVMMC::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ----------------------------------------------------------------------
   attempt Virtual Move Monte Carlo translations and rotations
   done before exchange, borders, reneighbor
   so that ghost atoms and neighbor lists will be correct
------------------------------------------------------------------------- */

void FixVMMC::pre_exchange()
{
  // just return if should not be called on this timestep

  if (next_reneighbor != update->ntimestep) return;

  mc_active = 1;

  double coordinates[domain->dimension*atom->natoms];
  double orientations[domain->dimension*atom->natoms];

  int i, ii, inum, id_vmmc;
  int *ilist;

  inum = list->inum; // number of atoms i for which neighbour lists are held
  ilist = list->ilist; // local index of atom i

  // copy particle coordinates and orientations
  for (ii=0; ii<inum; ii++) {

    i = ilist[ii]; // assign local index to i
    id_vmmc = atom->tag[i]-1; // work out libVMMC index

    coordinates[domain->dimension*id_vmmc + 0] = atom->x[i][0];
    coordinates[domain->dimension*id_vmmc + 1] = atom->x[i][1];
    coordinates[domain->dimension*id_vmmc + 2] = atom->x[i][2];

#ifdef DIPOLE
    orientations[domain->dimension*id_vmmc + 0] = atom->mu[i][0];
    orientations[domain->dimension*id_vmmc + 1] = atom->mu[i][1];
    orientations[domain->dimension*id_vmmc + 2] = atom->mu[i][2];
#endif
#ifndef DIPOLE
    orientations[domain->dimension*id_vmmc + 0] = 1.0;
    orientations[domain->dimension*id_vmmc + 1] = 0.0;
    orientations[domain->dimension*id_vmmc + 2] = 0.0;
#endif

  }

  vmmc->setPositions(coordinates);
  vmmc->setOrientations(orientations);

  // perform nvmmcmoves VMMC trial moves
  vmmc->step(nvmmcmoves);

  next_reneighbor = update->ntimestep + nevery;

  mc_active = 0;

}

/* ----------------------------------------------------------------------
   compute pair interaction between two particles for libVMMC library
------------------------------------------------------------------------- */

double FixVMMC::pair_energy(
    unsigned int index1, const double* pos1, const double* orient1,
    unsigned int index2, const double* pos2, const double* orient2){   
  
  int *type = atom->type;
  pair = force->pair;
  cutsq = force->pair->cutsq;
  double delr[3], rsq;

  double fpair = 0.0;
  double factor_coul = 1.0;
  double factor_lj = 1.0;
  double total_energy = 0.0;

  // local indices
  int i1 = atom->map(index1+1);
  int i2 = atom->map(index2+1);

  int i1type = type[i1];
  int i2type = type[i2];

  //calculate separation distance
  delr[0] = pos1[0]-pos2[0];
  delr[1] = pos1[1]-pos2[1];
  delr[2] = pos1[2]-pos2[2];

  // using libVMMC coordinates, hence enforce minimum image
  for (int ii=0; ii<3; ii++) {
    if (delr[ii] < -0.5*domain->boxhi[ii]) delr[ii] += domain->boxhi[ii];
    if (delr[ii] >= 0.5*domain->boxhi[ii]) delr[ii] -= domain->boxhi[ii];
  }

  rsq = delr[0]*delr[0] + delr[1]*delr[1] + delr[2]*delr[2];

#ifdef DIPOLE
  PairLJCutDipoleCut* ptr = dynamic_cast<PairLJCutDipoleCut*>(pair); // get a pointer to derived pair style
#endif
  // calculate pair energy if within cutoff
  if(rsq < cutsq[i1type][i2type]){
#ifdef DIPOLE
    total_energy = ptr->pair_energy_vmmc(i1,i2,i1type,i2type,delr,orient1,orient2);     // bespoke pair energy function 
#endif
#ifndef DIPOLE
    total_energy = pair->single(i1,i2,i1type,i2type,rsq,factor_coul,factor_lj,fpair); // default LAMMPS single routine
#endif
  }

  return total_energy;
}


/* ----------------------------------------------------------------------
   determine all interactions for a given particle for libVMMC library
------------------------------------------------------------------------- */

unsigned int FixVMMC::interactions(
    unsigned int index, const double* pos, const double* orient, unsigned int* interact)
{
  int i, j, ii, jj, inum, jnum;
  int *ilist, *jlist, *numneigh, **firstneigh;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  i = atom->map(index+1); // work out local index
  jnum = numneigh[i]; // determine number of neighbors
  jlist = firstneigh[i]; // pointer to first neighbor

  for (jj=0; jj<jnum; jj++) {

    j = jlist[jj];
    j &= NEIGHMASK;

    interact[jj]=atom->tag[j]-1; // work out libVMMC index and store

  }

  return jnum;
}

/* ----------------------------------------------------------------------
   post move update of atom coordinates and orientations
------------------------------------------------------------------------- */

void FixVMMC::post_move(
    unsigned int index, const double* pos, const double* orient)
{
  int i = atom->map(index+1); // work out local index

  // move libVMMC coordinates into coordinate array
  atom->x[i][0] = pos[0];
  atom->x[i][1] = pos[1];
  atom->x[i][2] = pos[2];

  // move libVMMC orientations into atom->mu array for stockmayer test
#ifdef DIPOLE 
  atom->mu[i][0] = orient[0];
  atom->mu[i][1] = orient[1];
  atom->mu[i][2] = orient[2];
#endif
  return;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixVMMC::memory_usage()
{
  double bytes = (double)vmmc_nmax * sizeof(int);
  return bytes;
}

/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void FixVMMC::write_restart(FILE *fp)
{
  int n = 0;
  double list[12];
  list[n++] = random_equal->state();
  list[n++] = random_unequal->state();
  list[n++] = ubuf(next_reneighbor).d;
  list[n++] = ubuf(update->ntimestep).d;

  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(list,sizeof(double),n,fp);
  }
}

/* ----------------------------------------------------------------------
   use state info from restart file to restart the Fix
------------------------------------------------------------------------- */

void FixVMMC::restart(char *buf)
{
  int n = 0;
  auto list = (double *) buf;

  seed = static_cast<int> (list[n++]);
  random_equal->reset(seed);

  seed = static_cast<int> (list[n++]);
  random_unequal->reset(seed);

  next_reneighbor = (bigint) ubuf(list[n++]).i;

  bigint ntimestep_restart = (bigint) ubuf(list[n++]).i;
  if (ntimestep_restart != update->ntimestep)
    error->all(FLERR,"Must not reset timestep when restarting fix vmmc");
}

/* ----------------------------------------------------------------------
   extract variable which stores whether MC is active or not
     active = MC moves are taking place
     not active = normal MD is taking place
   extract variable which stores index of exclusion group
------------------------------------------------------------------------- */

void *FixVMMC::extract(const char *name, int &dim)
{
  if (strcmp(name,"mc_active") == 0) {
    dim = 0;
    return (void *) &mc_active;
  }
  if (strcmp(name,"exclusion_group") == 0) {
    dim = 0;
    return (void *) &exclusion_group;
  }
  return nullptr;
}
