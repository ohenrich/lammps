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
   Contributing authors: Oliver Henrich (U Strathclye, Glasgow)
------------------------------------------------------------------------- */

#include "fix_langevin_wall.h"

#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "compute.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "random_mars.h"
#include "update.h"
#include "fix_wall.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

enum{NOBIAS,BIAS};
enum{CONSTANT,EQUAL,ATOM};

#define EINERTIA 0.2          // moment of inertia prefactor for ellipsoid

/* ---------------------------------------------------------------------- */

FixLangevinWall::FixLangevinWall(LAMMPS *lmp, int narg, char **arg) :
  FixLangevin(lmp, narg, arg)
{
  rad = new double[atom->ntypes+1];
}

/* ---------------------------------------------------------------------- */

FixLangevinWall::~FixLangevinWall()
{
  delete [] rad;
}

/* ---------------------------------------------------------------------- */

void FixLangevinWall::init()
{
  int nwallfix=0;

  FixLangevin::init();

  for (int i = 0; i < modify->nfix; i++) {
    if (strstr(modify->fix[i]->style,"wall/") != nullptr) {
      wallfix[nwallfix] = dynamic_cast<FixWall *>(modify->fix[i]);
      nwallfix++;
    }
  }

 // setting radii
  for (int n=1; n<atom->ntypes; n++)
    rad[n] = 1.0; // nucleotide radius = 1 l.u.
  // bead radius according to position of secondary wall
  rad[atom->ntypes] = wallfix[1]->coord0[0]+wallfix[1]->cutoff[0];
}

/* ---------------------------------------------------------------------- */

void FixLangevinWall::post_force(int /*vflag*/)
{
  double *rmass = atom->rmass;

  // enumerate all 2^6 possibilities for template parameters
  // this avoids testing them inside inner loop:
  // TSTYLEATOM, GJF, TALLY, BIAS, RMASS, ZERO

  if (tstyle == ATOM)
    if (gjfflag)
      if (tallyflag || osflag)
        if (tbiasflag == BIAS)
          if (rmass)
            if (zeroflag) post_force_templated<1,1,1,1,1,1>();
            else          post_force_templated<1,1,1,1,1,0>();
          else
            if (zeroflag) post_force_templated<1,1,1,1,0,1>();
            else          post_force_templated<1,1,1,1,0,0>();
        else
          if (rmass)
            if (zeroflag) post_force_templated<1,1,1,0,1,1>();
            else          post_force_templated<1,1,1,0,1,0>();
          else
            if (zeroflag) post_force_templated<1,1,1,0,0,1>();
            else          post_force_templated<1,1,1,0,0,0>();
      else
        if (tbiasflag == BIAS)
          if (rmass)
            if (zeroflag) post_force_templated<1,1,0,1,1,1>();
            else          post_force_templated<1,1,0,1,1,0>();
          else
            if (zeroflag) post_force_templated<1,1,0,1,0,1>();
            else          post_force_templated<1,1,0,1,0,0>();
        else
          if (rmass)
            if (zeroflag) post_force_templated<1,1,0,0,1,1>();
            else          post_force_templated<1,1,0,0,1,0>();
          else
            if (zeroflag) post_force_templated<1,1,0,0,0,1>();
            else          post_force_templated<1,1,0,0,0,0>();
    else
      if (tallyflag || osflag)
        if (tbiasflag == BIAS)
          if (rmass)
            if (zeroflag) post_force_templated<1,0,1,1,1,1>();
            else          post_force_templated<1,0,1,1,1,0>();
          else
            if (zeroflag) post_force_templated<1,0,1,1,0,1>();
            else          post_force_templated<1,0,1,1,0,0>();
        else
          if (rmass)
            if (zeroflag) post_force_templated<1,0,1,0,1,1>();
            else          post_force_templated<1,0,1,0,1,0>();
          else
            if (zeroflag) post_force_templated<1,0,1,0,0,1>();
            else          post_force_templated<1,0,1,0,0,0>();
      else
        if (tbiasflag == BIAS)
          if (rmass)
            if (zeroflag) post_force_templated<1,0,0,1,1,1>();
            else          post_force_templated<1,0,0,1,1,0>();
          else
            if (zeroflag) post_force_templated<1,0,0,1,0,1>();
            else          post_force_templated<1,0,0,1,0,0>();
        else
          if (rmass)
            if (zeroflag) post_force_templated<1,0,0,0,1,1>();
            else          post_force_templated<1,0,0,0,1,0>();
          else
            if (zeroflag) post_force_templated<1,0,0,0,0,1>();
            else          post_force_templated<1,0,0,0,0,0>();
  else
    if (gjfflag)
      if (tallyflag  || osflag)
        if (tbiasflag == BIAS)
          if (rmass)
            if (zeroflag) post_force_templated<0,1,1,1,1,1>();
            else          post_force_templated<0,1,1,1,1,0>();
          else
            if (zeroflag) post_force_templated<0,1,1,1,0,1>();
            else          post_force_templated<0,1,1,1,0,0>();
        else
          if (rmass)
            if (zeroflag) post_force_templated<0,1,1,0,1,1>();
            else          post_force_templated<0,1,1,0,1,0>();
          else
            if (zeroflag) post_force_templated<0,1,1,0,0,1>();
            else          post_force_templated<0,1,1,0,0,0>();
      else
        if (tbiasflag == BIAS)
          if (rmass)
            if (zeroflag) post_force_templated<0,1,0,1,1,1>();
            else          post_force_templated<0,1,0,1,1,0>();
          else
            if (zeroflag) post_force_templated<0,1,0,1,0,1>();
            else          post_force_templated<0,1,0,1,0,0>();
        else
          if (rmass)
            if (zeroflag) post_force_templated<0,1,0,0,1,1>();
            else          post_force_templated<0,1,0,0,1,0>();
          else
            if (zeroflag) post_force_templated<0,1,0,0,0,1>();
            else          post_force_templated<0,1,0,0,0,0>();
    else
      if (tallyflag || osflag)
        if (tbiasflag == BIAS)
          if (rmass)
            if (zeroflag) post_force_templated<0,0,1,1,1,1>();
            else          post_force_templated<0,0,1,1,1,0>();
          else
            if (zeroflag) post_force_templated<0,0,1,1,0,1>();
            else          post_force_templated<0,0,1,1,0,0>();
        else
          if (rmass)
            if (zeroflag) post_force_templated<0,0,1,0,1,1>();
            else          post_force_templated<0,0,1,0,1,0>();
          else
            if (zeroflag) post_force_templated<0,0,1,0,0,1>();
            else          post_force_templated<0,0,1,0,0,0>();
      else
        if (tbiasflag == BIAS)
          if (rmass)
            if (zeroflag) post_force_templated<0,0,0,1,1,1>();
            else          post_force_templated<0,0,0,1,1,0>();
          else
            if (zeroflag) post_force_templated<0,0,0,1,0,1>();
            else          post_force_templated<0,0,0,1,0,0>();
        else
          if (rmass)
            if (zeroflag) post_force_templated<0,0,0,0,1,1>();
            else          post_force_templated<0,0,0,0,1,0>();
          else
            if (zeroflag) post_force_templated<0,0,0,0,0,1>();
            else          post_force_templated<0,0,0,0,0,0>();
}

/* ----------------------------------------------------------------------
   modify forces using one of the many Langevin styles
------------------------------------------------------------------------- */

template < int Tp_TSTYLEATOM, int Tp_GJF, int Tp_TALLY,
           int Tp_BIAS, int Tp_RMASS, int Tp_ZERO >
void FixLangevinWall::post_force_templated()
{
  double gamma1,gamma2;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double aux,dhi[3],sqrtdhi[3];

  // apply damping and thermostat to atoms in group

  // for Tp_TSTYLEATOM:
  //   use per-atom per-coord target temperature
  // for Tp_GJF:
  //   use Gronbech-Jensen/Farago algorithm
  //   else use regular algorithm
  // for Tp_TALLY:
  //   store drag plus random forces in flangevin[nlocal][3]
  // for Tp_BIAS:
  //   calculate temperature since some computes require temp
  //   computed on current nlocal atoms to remove bias
  //   test v = 0 since some computes mask non-participating atoms via v = 0
  //   and added force has extra term not multiplied by v = 0
  // for Tp_RMASS:
  //   use per-atom masses
  //   else use per-type masses
  // for Tp_ZERO:
  //   sum random force over all atoms in group
  //   subtract sum/count from each atom in group

  double fdrag[3],fran[3],fsum[3],fsumall[3];
  bigint count;
  double fswap;

  double boltz = force->boltz;
  double dt = update->dt;
  double mvv2e = force->mvv2e;
  double ftm2v = force->ftm2v;

  compute_target();

  if (Tp_ZERO) {
    fsum[0] = fsum[1] = fsum[2] = 0.0;
    count = group->count(igroup);
    if (count == 0)
      error->all(FLERR,"Cannot zero Langevin force of 0 atoms");
  }

  // reallocate flangevin if necessary

  if (Tp_TALLY) {
    if (atom->nmax > maxatom1) {
      memory->destroy(flangevin);
      maxatom1 = atom->nmax;
      memory->create(flangevin,maxatom1,3,"langevin:flangevin");
    }
    flangevin_allocated = 1;
  }

  if (Tp_BIAS) temperature->compute_scalar();

  for (int i = 0; i < nlocal; i++) {

    // auxilliary factor: ratio radius to distance from main wall
    aux = rad[type[i]]/(x[i][2]-wallfix[0]->coord0[0]);

    // scaled diffusivity due to hydrodynamic interactions
    // parallel to the wall in x- and y-direction
    dhi[0] = 1.0 - 0.5625*aux + 0.125*aux*aux*aux;
    sqrtdhi[0] = sqrt(dhi[0]);

    dhi[1] = dhi[0];
    sqrtdhi[1] = sqrt(dhi[1]);

    // perpendicular to the wall in z-direction
    dhi[2] = 1.0 - 1.125*aux + 0.5*aux*aux*aux;
    sqrtdhi[2] = sqrt(dhi[2]);

    if (mask[i] & groupbit) {
      if (Tp_TSTYLEATOM) tsqrt = sqrt(tforce[i]);
      if (Tp_RMASS) {
        gamma1 = -rmass[i] / t_period / ftm2v;
        if (Tp_GJF)
          gamma2 = sqrt(rmass[i]) * sqrt(2.0*boltz/t_period/dt/mvv2e) / ftm2v;
        else
          gamma2 = sqrt(rmass[i]) * sqrt(24.0*boltz/t_period/dt/mvv2e) / ftm2v;
        gamma1 *= 1.0/ratio[type[i]];
        gamma2 *= 1.0/sqrt(ratio[type[i]]) * tsqrt;
      } else {
        gamma1 = gfactor1[type[i]];
        gamma2 = gfactor2[type[i]] * tsqrt;
      }

      if (Tp_GJF) {
        fran[0] = gamma2*random->gaussian()/sqrtdhi[0];
        fran[1] = gamma2*random->gaussian()/sqrtdhi[1];
        fran[2] = gamma2*random->gaussian()/sqrtdhi[2];
      } else {
        fran[0] = gamma2*(random->uniform()-0.5)/sqrtdhi[0];
        fran[1] = gamma2*(random->uniform()-0.5)/sqrtdhi[1];
        fran[2] = gamma2*(random->uniform()-0.5)/sqrtdhi[2];
      }

      if (Tp_BIAS) {
        temperature->remove_bias(i,v[i]);
        fdrag[0] = gamma1*v[i][0]/dhi[0];
        fdrag[1] = gamma1*v[i][1]/dhi[1];
        fdrag[2] = gamma1*v[i][2]/dhi[2];
        if (v[i][0] == 0.0) fran[0] = 0.0;
        if (v[i][1] == 0.0) fran[1] = 0.0;
        if (v[i][2] == 0.0) fran[2] = 0.0;
        temperature->restore_bias(i,v[i]);
      } else {
        fdrag[0] = gamma1*v[i][0]/dhi[0];
        fdrag[1] = gamma1*v[i][1]/dhi[1];
        fdrag[2] = gamma1*v[i][2]/dhi[2];
      }

      if (Tp_GJF) {
        if (Tp_BIAS)
          temperature->remove_bias(i,v[i]);
        lv[i][0] = gjfsib*v[i][0];
        lv[i][1] = gjfsib*v[i][1];
        lv[i][2] = gjfsib*v[i][2];
        if (Tp_BIAS)
          temperature->restore_bias(i,v[i]);
        if (Tp_BIAS)
          temperature->restore_bias(i,lv[i]);

        fswap = 0.5*(fran[0]+franprev[i][0]);
        franprev[i][0] = fran[0];
        fran[0] = fswap;
        fswap = 0.5*(fran[1]+franprev[i][1]);
        franprev[i][1] = fran[1];
        fran[1] = fswap;
        fswap = 0.5*(fran[2]+franprev[i][2]);
        franprev[i][2] = fran[2];
        fran[2] = fswap;

        fdrag[0] *= gjfa;
        fdrag[1] *= gjfa;
        fdrag[2] *= gjfa;
        fran[0] *= gjfa;
        fran[1] *= gjfa;
        fran[2] *= gjfa;
        f[i][0] *= gjfa;
        f[i][1] *= gjfa;
        f[i][2] *= gjfa;
      }

      f[i][0] += fdrag[0] + fran[0];
      f[i][1] += fdrag[1] + fran[1];
      f[i][2] += fdrag[2] + fran[2];

      if (Tp_ZERO) {
        fsum[0] += fran[0];
        fsum[1] += fran[1];
        fsum[2] += fran[2];
      }

      if (Tp_TALLY) {
        if (Tp_GJF) {
          fdrag[0] = gamma1*lv[i][0]/gjfsib/gjfsib/dhi[0];
          fdrag[1] = gamma1*lv[i][1]/gjfsib/gjfsib/dhi[1];
          fdrag[2] = gamma1*lv[i][2]/gjfsib/gjfsib/dhi[2];
          fswap = (2*fran[0]/gjfa - franprev[i][0])/gjfsib;
          fran[0] = fswap;
          fswap = (2*fran[1]/gjfa - franprev[i][1])/gjfsib;
          fran[1] = fswap;
          fswap = (2*fran[2]/gjfa - franprev[i][2])/gjfsib;
          fran[2] = fswap;
        }
        flangevin[i][0] = fdrag[0] + fran[0];
        flangevin[i][1] = fdrag[1] + fran[1];
        flangevin[i][2] = fdrag[2] + fran[2];

      }
    }
  }

  // set total force to zero

  if (Tp_ZERO) {
    MPI_Allreduce(fsum,fsumall,3,MPI_DOUBLE,MPI_SUM,world);
    fsumall[0] /= count;
    fsumall[1] /= count;
    fsumall[2] /= count;
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        f[i][0] -= fsumall[0];
        f[i][1] -= fsumall[1];
        f[i][2] -= fsumall[2];
        if (Tp_TALLY) {
          flangevin[i][0] -= fsumall[0];
          flangevin[i][1] -= fsumall[1];
          flangevin[i][2] -= fsumall[2];
        }
      }
    }
  }

  // thermostat angmom
  if (ascale) angmom_thermostat();
}

/* ----------------------------------------------------------------------
   thermostat rotational dof via angmom
------------------------------------------------------------------------- */

void FixLangevinWall::angmom_thermostat()
{
  double gamma1,gamma2;

  double boltz = force->boltz;
  double dt = update->dt;
  double mvv2e = force->mvv2e;
  double ftm2v = force->ftm2v;

  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  double **torque = atom->torque;
  double **angmom = atom->angmom;
  double *rmass = atom->rmass;
  int *ellipsoid = atom->ellipsoid;
  int *mask = atom->mask;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  // rescale gamma1/gamma2 by ascale for aspherical particles
  // does not affect rotational thermosatting
  // gives correct rotational diffusivity behavior if (nearly) spherical
  // any value will be incorrect for rotational diffusivity if aspherical

  double inertia[3],omega[3],tran[3];
  double *shape,*quat;

  double **x = atom->x;
  double aux,dhi[3],sqrtdhi[3];

  for (int i = 0; i < nlocal; i++) {

    // auxilliary factor: ratio radius to distance from main wall
    aux = rad[type[i]]/(x[i][2]-wallfix[0]->coord0[0]);
        
    // scaled diffusivity due to hydrodynamic interactions
    // parallel to the wall in x- and y-direction
    dhi[0] = 1.0 - 0.5625*aux + 0.125*aux*aux*aux;
    sqrtdhi[0] = sqrt(dhi[0]);
        
    dhi[1] = dhi[0];
    sqrtdhi[1] = sqrt(dhi[1]);
        
    // perpendicular to the wall in z-direction
    dhi[2] = 1.0 - 1.125*aux + 0.5*aux*aux*aux;
    sqrtdhi[2] = sqrt(dhi[2]);

    if (mask[i] & groupbit) {
      shape = bonus[ellipsoid[i]].shape;
      inertia[0] = EINERTIA*rmass[i] * (shape[1]*shape[1]+shape[2]*shape[2]);
      inertia[1] = EINERTIA*rmass[i] * (shape[0]*shape[0]+shape[2]*shape[2]);
      inertia[2] = EINERTIA*rmass[i] * (shape[0]*shape[0]+shape[1]*shape[1]);
      quat = bonus[ellipsoid[i]].quat;
      MathExtra::mq_to_omega(angmom[i],quat,inertia,omega);

      if (tstyle == ATOM) tsqrt = sqrt(tforce[i]);
      gamma1 = -ascale / t_period / ftm2v;
      gamma2 = sqrt(ascale*24.0*boltz/t_period/dt/mvv2e) / ftm2v;
      gamma1 *= 1.0/ratio[type[i]];
      gamma2 *= 1.0/sqrt(ratio[type[i]]) * tsqrt;
      tran[0] = sqrt(inertia[0])*gamma2*(random->uniform()-0.5)/sqrtdhi[0];
      tran[1] = sqrt(inertia[1])*gamma2*(random->uniform()-0.5)/sqrtdhi[1];
      tran[2] = sqrt(inertia[2])*gamma2*(random->uniform()-0.5)/sqrtdhi[2];
      torque[i][0] += inertia[0]*gamma1*omega[0]/dhi[0] + tran[0];
      torque[i][1] += inertia[1]*gamma1*omega[1]/dhi[1] + tran[1];
      torque[i][2] += inertia[2]*gamma1*omega[2]/dhi[2] + tran[2];
    }
  }
}
