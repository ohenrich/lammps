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

#ifndef MF_OXDNA_H
#define MF_OXDNA_H

#include "math_extra.h"

namespace MFOxdna {

inline double F1(double, double, double, double, double, double, double, double, double, double, double, double &);
inline double F2(double, double, double, double, double, double, double, double, double, double, double &);
inline double F3(double, double, double, double, double, double, double, double &);
inline double F4(double, double, double, double, double, double, double &);
inline double F5(double, double, double, double, double, double &);
inline double F6(double, double, double, double &);
inline double is_3pto5p(const double *, const double *);

}    // namespace MFOxdna

/* ----------------------------------------------------------------------
   f1 modulation factor
   ------------------------------------------------------------------------- */
inline double MFOxdna::F1(double r, double eps, double a, double cut_0, double cut_lc,
                          double cut_hc, double cut_lo, double cut_hi, double b_lo, double b_hi,
                          double shift, double &df)
{

  if (r > cut_hc) {
    df = 0.0;
    return 0.0;
  } else if (r > cut_hi) {
    df = 2 * eps * b_hi * (1 - cut_hc / r);
    return eps * b_hi * (r - cut_hc) * (r - cut_hc);
  } else if (r > cut_lo) {
    double tmp = 1 - exp(-(r - cut_0) * a);
    df = 2 * eps * (1 - tmp) * tmp * a / r;
    return eps * tmp * tmp - shift;
  } else if (r > cut_lc) {
    df = 2 * eps * b_lo * (1 - cut_lc / r);
    return eps * b_lo * (r - cut_lc) * (r - cut_lc);
  } else {
    df = 0.0;
    return 0.0;
  }
}

/* ----------------------------------------------------------------------
   f2 modulation factor
   ------------------------------------------------------------------------- */
inline double MFOxdna::F2(double r, double k, double cut_0, double cut_lc, double cut_hc,
                          double cut_lo, double cut_hi, double b_lo, double b_hi, double cut_c,
                          double &df)
{

  if (r < cut_lc || r > cut_hc) {
    df = 0.0;
    return 0.0;
  } else if (r < cut_lo) {
    df = 2 * k * b_lo * (r - cut_lc);
    return k * b_lo * (cut_lc - r) * (cut_lc - r);
  } else if (r < cut_hi) {
    df = k * (r - cut_0);
    return k * 0.5 * ((r - cut_0) * (r - cut_0) - (cut_0 - cut_c) * (cut_0 - cut_c));
  } else {
    df = 2 * k * b_hi * (r - cut_hc);
    return k * b_hi * (cut_hc - r) * (cut_hc - r);
  }
}

/* ----------------------------------------------------------------------
   f3 modulation factor
   ------------------------------------------------------------------------- */
inline double MFOxdna::F3(double rsq, double cutsq_ast, double cut_c, double lj1, double lj2,
                          double eps, double b, double &df)
{
  double evdwl = 0.0;

  if (rsq < cutsq_ast) {
    double r2inv = 1.0 / rsq;
    double r6inv = r2inv * r2inv * r2inv;
    df = r2inv * r6inv * (12 * lj1 * r6inv - 6 * lj2);
    evdwl = r6inv * (lj1 * r6inv - lj2);
  } else {
    double r = sqrt(rsq);
    double rinv = 1.0 / r;
    df = 2 * eps * b * (cut_c * rinv - 1);
    evdwl = eps * b * (cut_c - r) * (cut_c - r);
  }
  return evdwl;
}

/* ----------------------------------------------------------------------
   f4 modulation factor

   NOTE: We handle the sin(theta) factor from the partial derivative
   of d(cos(theta))/dtheta externally. The reason for this is
   because the sign of DF4 depends on the sign of theta in the
   function call. It is also more efficient to store sin(theta).
   ------------------------------------------------------------------------- */
inline double MFOxdna::F4(double theta, double a, double theta_0, double dtheta_ast, double b,
                          double dtheta_c, double &df)
{
  double dtheta = theta - theta_0;

  if (fabs(dtheta) > dtheta_c) {
    df = 0.0;
    return 0.0;
  } else if (dtheta > dtheta_ast) {
    df = 2 * b * (dtheta - dtheta_c); 
    return b * (dtheta - dtheta_c) * (dtheta - dtheta_c);
  } else if (dtheta > -dtheta_ast) {
    df = -2 * a * dtheta;
    return 1 - a * dtheta * dtheta;
  } else {
    df = 2 * b * (dtheta + dtheta_c);
    return b * (dtheta + dtheta_c) * (dtheta + dtheta_c);
  }
}

/* ----------------------------------------------------------------------
   f5 modulation factor
   ------------------------------------------------------------------------- */
inline double MFOxdna::F5(double x, double a, double x_ast, double b, double x_c, double &df)
{

  if (x >= 0) {
    df = 0.0;
    return 1.0;
  } else if (x > x_ast) {
    df = -2 * a * x;
    return 1 - a * x * x;
  } else if (x > x_c) {
    df = 2 * b * (x - x_c);
    return b * (x - x_c) * (x - x_c);
  } else {
    df = 0.0;
    return 0.0;
  }
}

/* ----------------------------------------------------------------------
   f6 modulation factor
   ------------------------------------------------------------------------- */
inline double MFOxdna::F6(double theta, double a, double b, double &df)
{
  if (theta < b) {
    df = 0.0;
    return 0.0;
  } else {
    df = a * (theta - b);
    return 0.5 * a * (theta - b) * (theta - b);
  }
}

/* ----------------------------------------------------------------------
   test for directionality by projecting base normal n onto delr = a - b,
   returns 1 if nucleotide b to nucleotide a is 3' to 5', otherwise -1
   ------------------------------------------------------------------------- */
inline double MFOxdna::is_3pto5p(const double *delr, const double *n)
{
  return copysign(1.0, MathExtra::dot3(delr, n));
}
#endif
