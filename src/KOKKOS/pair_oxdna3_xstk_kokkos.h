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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(oxdna3/xstk/kk,PairOxdna3XstkKokkos<LMPDeviceType>);
PairStyle(oxdna3/xstk/kk/device,PairOxdna3XstkKokkos<LMPDeviceType>);
PairStyle(oxdna3/xstk/kk/host,PairOxdna3XstkKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_OXDNA3_XSTK_KOKKOS_H
#define LMP_PAIR_OXDNA3_XSTK_KOKKOS_H

#include "kokkos_base.h"
#include "pair_kokkos.h"
#include "pair_oxdna3_xstk.h"
#include "neigh_list_kokkos.h"

#include "fix_oxdna_prime_neighs_kokkos.h"
#include "mf_oxdna_kokkos.h"

namespace LAMMPS_NS {

// Structure to hold all 2D cross-stacking parameters
struct ParamsXSTK {
  KK_FLOAT k_xst, b_xst_lo, b_xst_hi;
  KK_FLOAT a_xst1, theta_xst1_0, dtheta_xst1_ast, b_xst1, dtheta_xst1_c;
  KK_FLOAT a_xst2, theta_xst2_0, dtheta_xst2_ast, b_xst2, dtheta_xst2_c;
  KK_FLOAT a_xst3, theta_xst3_0, dtheta_xst3_ast, b_xst3, dtheta_xst3_c;
};

// Structure for 4D parameters (3'-3' interactions)
struct ParamsXSTK33 {
  KK_FLOAT cut_xst_0, cut_xst_c, cut_xst_lo, cut_xst_hi, cut_xst_lc, cut_xst_hc;
  KK_FLOAT a_xst4, theta_xst4_0, dtheta_xst4_ast, b_xst4, dtheta_xst4_c;
};

// Structure for 4D parameters (5'-5' interactions)
struct ParamsXSTK55 {
  KK_FLOAT cut_xst_0, cut_xst_c, cut_xst_lo, cut_xst_hi, cut_xst_lc, cut_xst_hc;
  KK_FLOAT a_xst4, theta_xst4_0, dtheta_xst4_ast, b_xst4, dtheta_xst4_c;
};

// Structure for theta7 parameters
struct ParamsXSTK7 {
  KK_FLOAT a_xst7, theta_xst7_0_33, theta_xst7_0_55, dtheta_xst7_ast, b_xst7, dtheta_xst7_c;
};

// Structure for theta8 parameters
struct ParamsXSTK8 {
  KK_FLOAT a_xst8, theta_xst8_0_33, theta_xst8_0_55, dtheta_xst8_ast, b_xst8, dtheta_xst8_c;
};

template<class DeviceType>
class FixOxdnaLRFKokkos;  // forward declaration

template<class DeviceType>
class FixOxdnaNpairKokkos;  // forward declaration

template<class DeviceType>
class FixOxdnaPrimeNeighsKokkos;  // forward declaration

template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
struct TagPairOxdna3XstkComputeNpair{};

template<class DeviceType>
class PairOxdna3XstkKokkos : public PairOxdna3Xstk, public KokkosBase {
 public:
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  PairOxdna3XstkKokkos(class LAMMPS *);
  ~PairOxdna3XstkKokkos() override;

  void compute(int, int) override;

  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  // Due to the need for pre-atom_mapping in KOKKOS which relies on a given neighbor list,
  // we do edge-based (oxdna npair) neigh list regardless of backend. So only one compute
  // kernel here.

  template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairOxdna3XstkComputeNpair<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int&, EV_FLOAT&) const;

  template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairOxdna3XstkComputeNpair<NEIGHFLAG,NEWTON_PAIR,EVFLAG>, const int&) const;

  template<int NEIGHFLAG, int NEWTON_PAIR>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void ev_tally_xyz(EV_FLOAT &ev, const int &i, const int &j,
      const KK_FLOAT &epair, const KK_ACC_FLOAT &fx, const KK_ACC_FLOAT &fy, const KK_ACC_FLOAT &fz,
      const KK_FLOAT &delx, const KK_FLOAT &dely, const KK_FLOAT &delz) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  int sbmask(const int& j) const;

 protected:

  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_kkacc_1d_3 f;
  typename AT::t_kkacc_1d_3 torque;
  typename AT::t_int_1d_randomread type;

  // Consolidated parameter views
  Kokkos::DualView<ParamsXSTK **, DeviceType> k_params_xstk;
  Kokkos::DualView<ParamsXSTK33 ****, DeviceType> k_params_33;
  Kokkos::DualView<ParamsXSTK55 ****, DeviceType> k_params_55;
  Kokkos::DualView<ParamsXSTK7 **, DeviceType> k_params_t7;
  Kokkos::DualView<ParamsXSTK8 **, DeviceType> k_params_t8;

  typename Kokkos::DualView<ParamsXSTK **, DeviceType>::t_dev_const_randomread d_params_xstk;
  typename Kokkos::DualView<ParamsXSTK33 ****, DeviceType>::t_dev_const_randomread d_params_33;
  typename Kokkos::DualView<ParamsXSTK55 ****, DeviceType>::t_dev_const_randomread d_params_55;
  typename Kokkos::DualView<ParamsXSTK7 **, DeviceType>::t_dev_const_randomread d_params_t7;
  typename Kokkos::DualView<ParamsXSTK8 **, DeviceType>::t_dev_const_randomread d_params_t8;

  DAT::ttransform_kkacc_1d k_eatom;
  DAT::ttransform_kkacc_1d_6 k_vatom;
  typename AT::t_kkacc_1d d_eatom;
  typename AT::t_kkacc_1d_6 d_vatom;

  int newton_pair;
  double special_lj[4];

  typename AT::tdual_kkfloat_2d k_cutsq;
  typename AT::t_kkfloat_2d d_cutsq;

  int neighflag;
  int nlocal, eflag, vflag;
  int anum;

  // Screening takes place on both Host/Device backends here - a design choice
  // to avoid the need for a second neighbor list and the associated overhead.
  // These are taken from the generic fix_oxdna_npairKK
  DAT::tdual_uint64_1d k_pairs_screened;
  typename AT::t_uint64_1d d_pairs_screened;
  int screened_pair_count;

  // cross-stacking interaction parameters
  typename AT::tdual_kkfloat_2d k_k_xst;
  typename AT::tdual_kkfloat_4d k_cut_xst_0_33, k_cut_xst_c_33, k_cut_xst_lo_33, k_cut_xst_hi_33;
  typename AT::tdual_kkfloat_4d k_cut_xst_lc_33, k_cut_xst_hc_33, k_cutsq_xst_hc_33;
  typename AT::tdual_kkfloat_4d k_cut_xst_0_55, k_cut_xst_c_55, k_cut_xst_lo_55, k_cut_xst_hi_55;
  typename AT::tdual_kkfloat_4d k_cut_xst_lc_55, k_cut_xst_hc_55, k_cutsq_xst_hc_55;
  typename AT::tdual_kkfloat_2d k_b_xst_lo, k_b_xst_hi;
  typename AT::tdual_kkfloat_2d k_a_xst1, k_theta_xst1_0, k_dtheta_xst1_ast;
  typename AT::tdual_kkfloat_2d k_b_xst1, k_dtheta_xst1_c;
  typename AT::tdual_kkfloat_2d k_a_xst2, k_theta_xst2_0, k_dtheta_xst2_ast;
  typename AT::tdual_kkfloat_2d k_b_xst2, k_dtheta_xst2_c;
  typename AT::tdual_kkfloat_2d k_a_xst3, k_theta_xst3_0, k_dtheta_xst3_ast;
  typename AT::tdual_kkfloat_2d k_b_xst3, k_dtheta_xst3_c;
  typename AT::tdual_kkfloat_4d k_a_xst4_33, k_theta_xst4_0_33, k_dtheta_xst4_ast_33;
  typename AT::tdual_kkfloat_4d k_b_xst4_33, k_dtheta_xst4_c_33;
  typename AT::tdual_kkfloat_4d k_a_xst4_55, k_theta_xst4_0_55, k_dtheta_xst4_ast_55;
  typename AT::tdual_kkfloat_4d k_b_xst4_55, k_dtheta_xst4_c_55;
  typename AT::tdual_kkfloat_2d k_a_xst7, k_theta_xst7_0_33, k_theta_xst7_0_55, k_dtheta_xst7_ast;
  typename AT::tdual_kkfloat_2d k_b_xst7, k_dtheta_xst7_c;
  typename AT::tdual_kkfloat_2d k_a_xst8, k_theta_xst8_0_33, k_theta_xst8_0_55, k_dtheta_xst8_ast;
  typename AT::tdual_kkfloat_2d k_b_xst8, k_dtheta_xst8_c;
  typename AT::t_kkfloat_2d_randomread d_k_xst;
  typename AT::t_kkfloat_4d_randomread d_cut_xst_0_33, d_cut_xst_c_33, d_cut_xst_lo_33, d_cut_xst_hi_33;
  typename AT::t_kkfloat_4d_randomread d_cut_xst_lc_33, d_cut_xst_hc_33, d_cutsq_xst_hc_33;
  typename AT::t_kkfloat_4d_randomread d_cut_xst_0_55, d_cut_xst_c_55, d_cut_xst_lo_55, d_cut_xst_hi_55;
  typename AT::t_kkfloat_4d_randomread d_cut_xst_lc_55, d_cut_xst_hc_55, d_cutsq_xst_hc_55;
  typename AT::t_kkfloat_2d_randomread d_b_xst_lo, d_b_xst_hi;
  typename AT::t_kkfloat_2d_randomread d_a_xst1, d_theta_xst1_0, d_dtheta_xst1_ast;
  typename AT::t_kkfloat_2d_randomread d_b_xst1, d_dtheta_xst1_c;
  typename AT::t_kkfloat_2d_randomread d_a_xst2, d_theta_xst2_0, d_dtheta_xst2_ast;
  typename AT::t_kkfloat_2d_randomread d_b_xst2, d_dtheta_xst2_c;
  typename AT::t_kkfloat_2d_randomread d_a_xst3, d_theta_xst3_0, d_dtheta_xst3_ast;
  typename AT::t_kkfloat_2d_randomread d_b_xst3, d_dtheta_xst3_c;
  typename AT::t_kkfloat_4d_randomread d_a_xst4_33, d_theta_xst4_0_33, d_dtheta_xst4_ast_33;
  typename AT::t_kkfloat_4d_randomread d_b_xst4_33, d_dtheta_xst4_c_33;
  typename AT::t_kkfloat_4d_randomread d_a_xst4_55, d_theta_xst4_0_55, d_dtheta_xst4_ast_55;
  typename AT::t_kkfloat_4d_randomread d_b_xst4_55, d_dtheta_xst4_c_55;
  typename AT::t_kkfloat_2d_randomread d_a_xst7, d_theta_xst7_0_33, d_theta_xst7_0_55, d_dtheta_xst7_ast;
  typename AT::t_kkfloat_2d_randomread d_b_xst7, d_dtheta_xst7_c;
  typename AT::t_kkfloat_2d_randomread d_a_xst8, d_theta_xst8_0_33, d_theta_xst8_0_55, d_dtheta_xst8_ast;
  typename AT::t_kkfloat_2d_randomread d_b_xst8, d_dtheta_xst8_c;
  // per-atom arrays for local unit vectors
  DAT::tdual_kkfloat_1d_3 k_nx_xtrct, k_ny_xtrct, k_nz_xtrct;
  typename AT::t_kkfloat_1d_3_randomread d_nx_xtrct, d_ny_xtrct, d_nz_xtrct;

  using KKDeviceType = typename KKDevice<DeviceType>::value;

  template<typename DataType, typename Layout>
  using DupScatterView = KKScatterView<DataType, Layout, KKDeviceType, \
  KKScatterSum, KKScatterDuplicated>;

  template<typename DataType, typename Layout>
  using NonDupScatterView = KKScatterView<DataType, Layout, KKDeviceType, \
  KKScatterSum, KKScatterNonDuplicated>;

  DupScatterView<KK_ACC_FLOAT*[3], typename AT::t_kkacc_1d_3::array_layout> dup_f;
  DupScatterView<KK_ACC_FLOAT*[3], typename AT::t_kkacc_1d_3::array_layout> dup_torque;
  DupScatterView<KK_ACC_FLOAT*, typename DAT::t_kkacc_1d::array_layout> dup_eatom;
  DupScatterView<KK_ACC_FLOAT*[6], typename DAT::t_kkacc_1d_6::array_layout> dup_vatom;
  NonDupScatterView<KK_ACC_FLOAT*[3], typename AT::t_kkacc_1d_3::array_layout> ndup_f;
  NonDupScatterView<KK_ACC_FLOAT*[3], typename AT::t_kkacc_1d_3::array_layout> ndup_torque;
  NonDupScatterView<KK_ACC_FLOAT*, typename DAT::t_kkacc_1d::array_layout> ndup_eatom;
  NonDupScatterView<KK_ACC_FLOAT*[6], typename DAT::t_kkacc_1d_6::array_layout> ndup_vatom;

  void allocate() override;
 
  friend void pair_virial_fdotr_compute<PairOxdna3XstkKokkos>(PairOxdna3XstkKokkos*);

  FixOxdnaLRFKokkos<DeviceType> *fix_oxdna_lrfKK;    // ptr to OXDNA/LRF/kk fix
  FixOxdnaNpairKokkos<DeviceType> *fix_oxdna_npairKK;    // ptr to OXDNA/NPAIR/kk fix
  FixOxdnaPrimeNeighsKokkos<DeviceType> *fix_oxdna_prime_neighsKK;    // ptr to OXDNA/PRIME_NEIGHS/kk fix
  bigint last_prime_neighs_xstk3_lastcall;
  typename AT::t_int_2d d_prime_neighs_oxdna3_xstk;

 private:

// The following is totally wild code-readability wise and I don't really like.
// But I was getting a ton of Live Register Pressure and the only way I could
// reduce this was to pull everything out into separate:
// PairOxdnaXstkKokkos<DeviceType>::xstk_* KOKKOS_INLINE_FUNCTIONs.
// The compilers (HIP and CUDA) wouldn't kill off short-lived vars otherwise,
// which really bumped up register usage and bumped up runtime. Simple INLINES
// didn't help.
//
// Compute-wise, it would be nice to calc the derivatives only when they are needed
// after the evdwl. But then I need to have my p_* terms again and the register pressure
// blows through occupancy and runtime goes up again. In these closed-scope areas,
// I've so found it best to just take the FP hit and calc the derivs even if they
// end up not being needed. So far, this is the fastest option.

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  bool xstk_radial_terms(const int &atype, const int &btype,
    const int &a3ptype, const int &a5ptype,
    const int &b3ptype, const int &b5ptype,
    const KK_FLOAT &r_hb,
    KK_FLOAT &f2_33, KK_FLOAT &f2_55,
    KK_FLOAT &df2_33, KK_FLOAT &df2_55) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  bool xstk_theta1_terms(const int &atype, const int &btype,
    const KK_FLOAT (&a_nx)[3], const KK_FLOAT (&b_nx)[3],
    KK_FLOAT &theta1, KK_FLOAT &f4t1, KK_FLOAT &df4t1) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  bool xstk_theta2_terms(const int &atype, const int &btype,
    const KK_FLOAT (&a_nx)[3], const KK_FLOAT (&delr_hb_norm)[3],
    KK_FLOAT &theta2, KK_FLOAT &cost2, KK_FLOAT &f4t2, KK_FLOAT &df4t2) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  bool xstk_theta3_terms(const int &atype, const int &btype,
    const KK_FLOAT (&b_nx)[3], const KK_FLOAT (&delr_hb_norm)[3],
    KK_FLOAT &theta3, KK_FLOAT &cost3, KK_FLOAT &f4t3, KK_FLOAT &df4t3) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  bool xstk_theta4_terms(const int &atype, const int &btype,
    const int &a3ptype, const int &a5ptype,
    const int &b3ptype, const int &b5ptype,
    const KK_FLOAT (&a_nz)[3], const KK_FLOAT (&b_nz)[3],
    KK_FLOAT &theta4,
    KK_FLOAT &f4t4_33, KK_FLOAT &f4t4_55,
    KK_FLOAT &df4t4_33, KK_FLOAT &df4t4_55) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  bool xstk_theta7_terms(const int &atype, const int &btype,
    const KK_FLOAT (&a_nz)[3], const KK_FLOAT (&delr_hb_norm)[3],
    KK_FLOAT &theta7, KK_FLOAT &cost7,
    KK_FLOAT &f4t7_33, KK_FLOAT &f4t7_55,
    KK_FLOAT &df4t7_33, KK_FLOAT &df4t7_55) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  bool xstk_theta8_terms(const int &atype, const int &btype,
    const KK_FLOAT (&b_nz)[3], const KK_FLOAT (&delr_hb_norm)[3],
    KK_FLOAT &theta8, KK_FLOAT &cost8,
    KK_FLOAT &f4t8_33, KK_FLOAT &f4t8_55,
    KK_FLOAT &df4t8_33, KK_FLOAT &df4t8_55) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void xstk_force_contrib(const KK_FLOAT &f2_33, const KK_FLOAT &f2_55,
    const KK_FLOAT &f4t1, const KK_FLOAT &f4t2, const KK_FLOAT &f4t3,
    const KK_FLOAT &f4t4_33, const KK_FLOAT &f4t4_55,
    const KK_FLOAT &f4t7_33, const KK_FLOAT &f4t7_55,
    const KK_FLOAT &f4t8_33, const KK_FLOAT &f4t8_55,
    const KK_FLOAT &df2_33, const KK_FLOAT &df2_55,
    const KK_FLOAT &df4t2, const KK_FLOAT &df4t3,
    const KK_FLOAT &df4t7_33, const KK_FLOAT &df4t7_55,
    const KK_FLOAT &df4t8_33, const KK_FLOAT &df4t8_55,
    const KK_FLOAT &rinv_hb, const KK_FLOAT &factor_lj,
    const KK_FLOAT &theta2, const KK_FLOAT &theta3, const KK_FLOAT &theta7, const KK_FLOAT &theta8,
    const KK_FLOAT &cost2, const KK_FLOAT &cost3, const KK_FLOAT &cost7, const KK_FLOAT &cost8,
    const KK_FLOAT (&delr_hb)[3], const KK_FLOAT (&delr_hb_norm)[3],
    const KK_FLOAT (&a_nx)[3], const KK_FLOAT (&b_nx)[3],
    const KK_FLOAT (&a_nz)[3], const KK_FLOAT (&b_nz)[3],
    const KK_FLOAT (&ra_chb)[3], const KK_FLOAT (&rb_chb)[3],
    KK_ACC_FLOAT (&delf)[3], KK_ACC_FLOAT (&delta)[3], KK_ACC_FLOAT (&deltb)[3]) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void xstk_torque_contrib(const KK_FLOAT &f2_33, const KK_FLOAT &f2_55,
    const KK_FLOAT &f4t1, const KK_FLOAT &f4t2, const KK_FLOAT &f4t3,
    const KK_FLOAT &f4t4_33, const KK_FLOAT &f4t4_55,
    const KK_FLOAT &f4t7_33, const KK_FLOAT &f4t7_55,
    const KK_FLOAT &f4t8_33, const KK_FLOAT &f4t8_55,
    const KK_FLOAT &df4t1, const KK_FLOAT &df4t2, const KK_FLOAT &df4t3,
    const KK_FLOAT &df4t4_33, const KK_FLOAT &df4t4_55,
    const KK_FLOAT &df4t7_33, const KK_FLOAT &df4t7_55,
    const KK_FLOAT &df4t8_33, const KK_FLOAT &df4t8_55,
    const KK_FLOAT &factor_lj,
    const KK_FLOAT &theta1, const KK_FLOAT &theta2, const KK_FLOAT &theta3,
    const KK_FLOAT &theta4,
    const KK_FLOAT &theta7, const KK_FLOAT &theta8,
    const KK_FLOAT (&a_nx)[3], const KK_FLOAT (&b_nx)[3],
    const KK_FLOAT (&a_nz)[3], const KK_FLOAT (&b_nz)[3],
    const KK_FLOAT (&delr_hb_norm)[3],
    KK_ACC_FLOAT (&delta)[3], KK_ACC_FLOAT (&deltb)[3]) const;
};

}

#endif
#endif
