// RUN: %clang_cc1 -faltivec -triple powerpc-unknown-unknown -emit-llvm %s -o - | FileCheck %s

#include "altivec.h"

int main ()
{
  vector signed char vsc = { 1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16 };
  vector unsigned char vuc = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
  vector short vs = { -1, 2, -3, 4, -5, 6, -7, 8 };
  vector unsigned short vus = { 1, 2, 3, 4, 5, 6, 7, 8 };
  vector int vi = { -1, 2, -3, 4 };
  vector unsigned int vui = { 1, 2, 3, 4 };
  vector float vf = { -1.5, 2.5, -3.5, 4.5 };

  vector signed char res_vsc;
  vector unsigned char res_vuc;
  vector short res_vs;
  vector unsigned short res_vus;
  vector int res_vi;
  vector unsigned int res_vui;
  vector float res_vf;

  int param_i;
  int res_i;

  /* vec_abs */
  vsc = vec_abs(vsc);              // CHECK: sub <16 x i8> zeroinitializer
                                   // CHECK: @llvm.ppc.altivec.vmaxsb

  vs = __builtin_vec_abs(vs);      // CHECK: sub <8 x i16> zeroinitializer
                                   // CHECK: @llvm.ppc.altivec.vmaxsh

  vi = vec_abs(vi);                // CHECK: sub <4 x i32> zeroinitializer
                                   // CHECK: @llvm.ppc.altivec.vmaxsw

  vf = vec_abs(vf);                // CHECK: store <4 x i32> <i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647>
                                   // CHECK: and <4 x i32>

  /* vec_abs */
  vsc = vec_abss(vsc);             // CHECK: @llvm.ppc.altivec.vsubsbs
                                   // CHECK: @llvm.ppc.altivec.vmaxsb

  vs = __builtin_vec_abss(vs);     // CHECK: @llvm.ppc.altivec.vsubshs
                                   // CHECK: @llvm.ppc.altivec.vmaxsh

  vi = vec_abss(vi);               // CHECK: @llvm.ppc.altivec.vsubsws
                                   // CHECK: @llvm.ppc.altivec.vmaxsw

  /*  vec_add */
  res_vsc  = vec_add(vsc, vsc);                 // CHECK: add nsw <16 x i8>
  res_vuc = vec_vaddubm(vuc, vuc);              // CHECK: add <16 x i8>
  res_vs  = __builtin_altivec_vadduhm(vs, vs);  // CHECK: add nsw <8 x i16>
  res_vus = vec_vadduhm(vus, vus);              // CHECK: add <8 x i16>
  res_vi  = __builtin_vec_vadduwm(vi, vi);      // CHECK: add nsw <4 x i32>
  res_vui = vec_vadduwm(vui, vui);              // CHECK: add <4 x i32>
  res_vf  = __builtin_vec_vaddfp(vf, vf);       // CHECK: fadd <4 x float>

  /* vec_addc */
  res_vui = vec_vaddcuw(vui, vui);              // HECK: @llvm.ppc.altivec.vaddcuw

  /* vec_adds */
  res_vsc  = vec_adds(vsc, vsc);                // CHECK: @llvm.ppc.altivec.vaddsbs
  res_vuc = vec_vaddubs(vuc, vuc);              // CHECK: @llvm.ppc.altivec.vaddubs
  res_vs  = __builtin_vec_vaddshs(vs, vs);      // CHECK: @llvm.ppc.altivec.vaddshs
  res_vus = vec_vadduhs(vus, vus);              // CHECK: @llvm.ppc.altivec.vadduhs
  res_vi  = __builtin_vec_vaddsws(vi, vi);      // CHECK: @llvm.ppc.altivec.vaddsws
  res_vui = vec_vadduws(vui, vui);              // CHECK: @llvm.ppc.altivec.vadduws

  /* vec_sub */
  res_vsc  = vec_sub(vsc, vsc);                 // CHECK: sub nsw <16 x i8>
  res_vuc = vec_vsububm(vuc, vuc);              // CHECK: sub <16 x i8>
  res_vs  = __builtin_altivec_vsubuhm(vs, vs);  // CHECK: sub nsw <8 x i16>
  res_vus = vec_vsubuhm(vus, vus);              // CHECK: sub <8 x i16>
  res_vi  = __builtin_vec_vsubuwm(vi, vi);      // CHECK: sub nsw <4 x i32>
  res_vui = vec_vsubuwm(vui, vui);              // CHECK: sub <4 x i32>
  res_vf  = __builtin_vec_vsubfp(vf, vf);       // CHECK: fsub <4 x float>

  /* vec_subs */
  res_vsc  = vec_subs(vsc, vsc);                // CHECK: @llvm.ppc.altivec.vsubsbs
  res_vuc = vec_vsububs(vuc, vuc);              // CHECK: @llvm.ppc.altivec.vsububs
  res_vs  = __builtin_vec_vsubshs(vs, vs);      // CHECK: @llvm.ppc.altivec.vsubshs
  res_vus = vec_vsubuhs(vus, vus);              // CHECK: @llvm.ppc.altivec.vsubuhs
  res_vi  = __builtin_vec_vsubsws(vi, vi);      // CHECK: @llvm.ppc.altivec.vsubsws
  res_vui = vec_vsubuws(vui, vui);              // CHECK: @llvm.ppc.altivec.vsubuws

  /* vec_avg */
  res_vsc  = vec_avg(vsc, vsc);                 // CHECK: @llvm.ppc.altivec.vavgsb
  res_vuc = __builtin_vec_vavgub(vuc, vuc);     // CHECK: @llvm.ppc.altivec.vavgub
  res_vs  = vec_vavgsh(vs, vs);                 // CHECK: @llvm.ppc.altivec.vavgsh
  res_vus = __builtin_vec_vavguh(vus, vus);     // CHECK: @llvm.ppc.altivec.vavguh
  res_vi  = vec_vavgsw(vi, vi);                 // CHECK: @llvm.ppc.altivec.vavgsw
  res_vui = __builtin_vec_vavguw(vui, vui);     // CHECK: @llvm.ppc.altivec.vavguw

  /* vec_st */
  param_i = 5;
  vec_st(vsc, 0, &res_vsc);                     // CHECK: @llvm.ppc.altivec.stvx
  __builtin_vec_st(vuc, param_i, &res_vuc);     // CHECK: @llvm.ppc.altivec.stvx
  vec_stvx(vs, 1, &res_vs);                     // CHECK: @llvm.ppc.altivec.stvx
  vec_st(vus, 1000, &res_vus);                  // CHECK: @llvm.ppc.altivec.stvx
  vec_st(vi, 0, &res_vi);                       // CHECK: @llvm.ppc.altivec.stvx
  vec_st(vui, 0, &res_vui);                     // CHECK: @llvm.ppc.altivec.stvx
  vec_st(vf, 0, &res_vf);                       // CHECK: @llvm.ppc.altivec.stvx

  /* vec_stl */
  param_i = 10000;
  vec_stl(vsc, param_i, &res_vsc);              // CHECK: @llvm.ppc.altivec.stvxl
  __builtin_vec_stl(vuc, 1, &res_vuc);          // CHECK: @llvm.ppc.altivec.stvxl
  vec_stvxl(vs, 0, &res_vs);                    // CHECK: @llvm.ppc.altivec.stvxl
  vec_stl(vus, 0, &res_vus);                    // CHECK: @llvm.ppc.altivec.stvxl
  vec_stl(vi, 0, &res_vi);                      // CHECK: @llvm.ppc.altivec.stvxl
  vec_stl(vui, 0, &res_vui);                    // CHECK: @llvm.ppc.altivec.stvxl
  vec_stl(vf, 0, &res_vf);                      // CHECK: @llvm.ppc.altivec.stvxl

  /* vec_ste */
  param_i = 10000;
  vec_ste(vsc, param_i, &res_vsc);              // CHECK: @llvm.ppc.altivec.stvebx
  vec_stvebx(vuc, 1, &res_vuc);                 // CHECK: @llvm.ppc.altivec.stvebx
  __builtin_vec_stvehx(vs, 0, &res_vs);         // CHECK: @llvm.ppc.altivec.stvehx
  vec_stvehx(vus, 0, &res_vus);                 // CHECK: @llvm.ppc.altivec.stvehx
  vec_stvewx(vi, 0, &res_vi);                   // CHECK: @llvm.ppc.altivec.stvewx
  __builtin_vec_stvewx(vui, 0, &res_vui);       // CHECK: @llvm.ppc.altivec.stvewx
  vec_stvewx(vf, 0, &res_vf);                   // CHECK: @llvm.ppc.altivec.stvewx

  /* vec_cmpb */
  res_vi = vec_vcmpbfp(vf, vf);                 // CHECK: @llvm.ppc.altivec.vcmpbfp

  /* vec_cmpeq */
  res_vi = vec_cmpeq(vsc, vsc);                 // CHECK: @llvm.ppc.altivec.vcmpequb
  res_vi = __builtin_vec_cmpeq(vuc, vuc);       // CHECK: @llvm.ppc.altivec.vcmpequb
  res_vi = vec_cmpeq(vs, vs);                   // CHECK: @llvm.ppc.altivec.vcmpequh
  res_vi = vec_cmpeq(vus, vus);                 // CHECK: @llvm.ppc.altivec.vcmpequh
  res_vi = vec_cmpeq(vi, vi);                   // CHECK: @llvm.ppc.altivec.vcmpequw
  res_vi = vec_cmpeq(vui, vui);                 // CHECK: @llvm.ppc.altivec.vcmpequw
  res_vi = vec_cmpeq(vf, vf);                   // CHECK: @llvm.ppc.altivec.vcmpeqfp

  /* vec_cmpge */
  res_vi = __builtin_vec_cmpge(vf, vf);         // CHECK: @llvm.ppc.altivec.vcmpgefp

  /* vec_cmpgt */
  res_vi = vec_cmpgt(vsc, vsc);                 // CHECK: @llvm.ppc.altivec.vcmpgtsb
  res_vi = vec_vcmpgtub(vuc, vuc);              // CHECK: @llvm.ppc.altivec.vcmpgtub
  res_vi = __builtin_vec_vcmpgtsh(vs, vs);      // CHECK: @llvm.ppc.altivec.vcmpgtsh
  res_vi = vec_cmpgt(vus, vus);                 // CHECK: @llvm.ppc.altivec.vcmpgtuh
  res_vi = vec_cmpgt(vi, vi);                   // CHECK: @llvm.ppc.altivec.vcmpgtsw
  res_vi = vec_cmpgt(vui, vui);                 // CHECK: @llvm.ppc.altivec.vcmpgtuw
  res_vi = vec_cmpgt(vf, vf);                   // CHECK: @llvm.ppc.altivec.vcmpgtfp

  /* vec_cmple */
  res_vi = __builtin_vec_cmple(vf, vf);         // CHECK: @llvm.ppc.altivec.vcmpgefp

  /* vec_cmplt */
  res_vi = vec_cmplt(vsc, vsc);                 // CHECK: @llvm.ppc.altivec.vcmpgtsb
  res_vi = __builtin_vec_cmplt(vuc, vuc);       // CHECK: @llvm.ppc.altivec.vcmpgtub
  res_vi = vec_cmplt(vs, vs);                   // CHECK: @llvm.ppc.altivec.vcmpgtsh
  res_vi = vec_cmplt(vus, vus);                 // CHECK: @llvm.ppc.altivec.vcmpgtuh
  res_vi = vec_cmplt(vi, vi);                   // CHECK: @llvm.ppc.altivec.vcmpgtsw
  res_vi = vec_cmplt(vui, vui);                 // CHECK: @llvm.ppc.altivec.vcmpgtuw
  res_vi = vec_cmplt(vf, vf);                   // CHECK: @llvm.ppc.altivec.vcmpgtfp

  /* vec_max */
  res_vsc  = vec_max(vsc, vsc);                 // CHECK: @llvm.ppc.altivec.vmaxsb
  res_vuc = __builtin_vec_vmaxub(vuc, vuc);     // CHECK: @llvm.ppc.altivec.vmaxub
  res_vs  = vec_vmaxsh(vs, vs);                 // CHECK: @llvm.ppc.altivec.vmaxsh
  res_vus = vec_max(vus, vus);                  // CHECK: @llvm.ppc.altivec.vmaxuh
  res_vi  = __builtin_vec_vmaxsw(vi, vi);       // CHECK: @llvm.ppc.altivec.vmaxsw
  res_vui = vec_vmaxuw(vui, vui);               // CHECK: @llvm.ppc.altivec.vmaxuw
  res_vf  = __builtin_vec_max(vf, vf);          // CHECK: @llvm.ppc.altivec.vmaxfp

  /* vec_mfvscr */
  vf = vec_mfvscr();                            // CHECK: @llvm.ppc.altivec.mfvscr

  /* vec_min */
  res_vsc  = vec_min(vsc, vsc);                 // CHECK: @llvm.ppc.altivec.vminsb
  res_vuc = __builtin_vec_vminub(vuc, vuc);     // CHECK: @llvm.ppc.altivec.vminub
  res_vs  = vec_vminsh(vs, vs);                 // CHECK: @llvm.ppc.altivec.vminsh
  res_vus = vec_min(vus, vus);                  // CHECK: @llvm.ppc.altivec.vminuh
  res_vi  = __builtin_vec_vminsw(vi, vi);       // CHECK: @llvm.ppc.altivec.vminsw
  res_vui = vec_vminuw(vui, vui);               // CHECK: @llvm.ppc.altivec.vminuw
  res_vf  = __builtin_vec_min(vf, vf);          // CHECK: @llvm.ppc.altivec.vminfp

  /* vec_mtvscr */
  vec_mtvscr(vsc);                              // CHECK: @llvm.ppc.altivec.mtvscr

  /* ------------------------------ predicates -------------------------------------- */

  res_i = __builtin_vec_vcmpeq_p(__CR6_EQ, vsc, vui); // CHECK: @llvm.ppc.altivec.vcmpeqfp.p
  res_i = __builtin_vec_vcmpge_p(__CR6_EQ, vs, vi);   // CHECK: @llvm.ppc.altivec.vcmpgefp.p
  res_i = __builtin_vec_vcmpgt_p(__CR6_EQ, vuc, vf);  // CHECK: @llvm.ppc.altivec.vcmpgtfp.p

  /*  vec_all_eq */
  res_i = vec_all_eq(vsc, vsc);                 // CHECK: @llvm.ppc.altivec.vcmpequb.p
  res_i = vec_all_eq(vuc, vuc);                 // CHECK: @llvm.ppc.altivec.vcmpequb.p
  res_i = vec_all_eq(vs, vs);                   // CHECK: @llvm.ppc.altivec.vcmpequh.p
  res_i = vec_all_eq(vus, vus);                 // CHECK: @llvm.ppc.altivec.vcmpequh.p
  res_i = vec_all_eq(vi, vi);                   // CHECK: @llvm.ppc.altivec.vcmpequw.p
  res_i = vec_all_eq(vui, vui);                 // CHECK: @llvm.ppc.altivec.vcmpequw.p
  res_i = vec_all_eq(vf, vf);                   // CHECK: @llvm.ppc.altivec.vcmpeqfp.p

  /* vec_all_ge */
  res_i = vec_all_ge(vsc, vsc);                 // CHECK: @llvm.ppc.altivec.vcmpgtsb.p
  res_i = vec_all_ge(vuc, vuc);                 // CHECK: @llvm.ppc.altivec.vcmpgtub.p
  res_i = vec_all_ge(vs, vs);                   // CHECK: @llvm.ppc.altivec.vcmpgtsh.p
  res_i = vec_all_ge(vus, vus);                 // CHECK: @llvm.ppc.altivec.vcmpgtuh.p
  res_i = vec_all_ge(vi, vi);                   // CHECK: @llvm.ppc.altivec.vcmpgtsw.p
  res_i = vec_all_ge(vui, vui);                 // CHECK: @llvm.ppc.altivec.vcmpgtuw.p
  res_i = vec_all_ge(vf, vf);                   // CHECK: @llvm.ppc.altivec.vcmpgtfp.p

  /* vec_all_gt */
  res_i = vec_all_gt(vsc, vsc);                 // CHECK: @llvm.ppc.altivec.vcmpgtsb.p
  res_i = vec_all_gt(vuc, vuc);                 // CHECK: @llvm.ppc.altivec.vcmpgtub.p
  res_i = vec_all_gt(vs, vs);                   // CHECK: @llvm.ppc.altivec.vcmpgtsh.p
  res_i = vec_all_gt(vus, vus);                 // CHECK: @llvm.ppc.altivec.vcmpgtuh.p
  res_i = vec_all_gt(vi, vi);                   // CHECK: @llvm.ppc.altivec.vcmpgtsw.p
  res_i = vec_all_gt(vui, vui);                 // CHECK: @llvm.ppc.altivec.vcmpgtuw.p
  res_i = vec_all_gt(vf, vf);                   // CHECK: @llvm.ppc.altivec.vcmpgtfp.p

  /* vec_all_in */
  res_i = vec_all_in(vf, vf);                   // CHECK: @llvm.ppc.altivec.vcmpbfp.p

  /* vec_all_le */
  res_i = vec_all_le(vsc, vsc);                 // CHECK: @llvm.ppc.altivec.vcmpgtsb.p
  res_i = vec_all_le(vuc, vuc);                 // CHECK: @llvm.ppc.altivec.vcmpgtub.p
  res_i = vec_all_le(vs, vs);                   // CHECK: @llvm.ppc.altivec.vcmpgtsh.p
  res_i = vec_all_le(vus, vus);                 // CHECK: @llvm.ppc.altivec.vcmpgtuh.p
  res_i = vec_all_le(vi, vi);                   // CHECK: @llvm.ppc.altivec.vcmpgtsw.p
  res_i = vec_all_le(vui, vui);                 // CHECK: @llvm.ppc.altivec.vcmpgtuw.p
  res_i = vec_all_le(vf, vf);                   // CHECK: @llvm.ppc.altivec.vcmpgtfp.p

  /* vec_all_nan */
  res_i = vec_all_nan(vf);                      // CHECK: @llvm.ppc.altivec.vcmpeqfp.p

  /*  vec_all_ne */
  res_i = vec_all_ne(vsc, vsc);                 // CHECK: @llvm.ppc.altivec.vcmpequb.p
  res_i = vec_all_ne(vuc, vuc);                 // CHECK: @llvm.ppc.altivec.vcmpequb.p
  res_i = vec_all_ne(vs, vs);                   // CHECK: @llvm.ppc.altivec.vcmpequh.p
  res_i = vec_all_ne(vus, vus);                 // CHECK: @llvm.ppc.altivec.vcmpequh.p
  res_i = vec_all_ne(vi, vi);                   // CHECK: @llvm.ppc.altivec.vcmpequw.p
  res_i = vec_all_ne(vui, vui);                 // CHECK: @llvm.ppc.altivec.vcmpequw.p
  res_i = vec_all_ne(vf, vf);                   // CHECK: @llvm.ppc.altivec.vcmpeqfp.p

  /* vec_all_nge */
  res_i = vec_all_nge(vf, vf);                  // CHECK: @llvm.ppc.altivec.vcmpgefp.p

  /* vec_all_ngt */
  res_i = vec_all_ngt(vf, vf);                  // CHECK: @llvm.ppc.altivec.vcmpgtfp.p

  /* vec_all_nle */
  res_i = vec_all_nle(vf, vf);                  // CHECK: @llvm.ppc.altivec.vcmpgefp.p

  /* vec_all_nlt */
  res_i = vec_all_nlt(vf, vf);                  // CHECK: @llvm.ppc.altivec.vcmpgtfp.p

  /* vec_all_numeric */
  res_i = vec_all_numeric(vf);                  // CHECK: @llvm.ppc.altivec.vcmpeqfp.p

  /*  vec_any_eq */
  res_i = vec_any_eq(vsc, vsc);                 // CHECK: @llvm.ppc.altivec.vcmpequb.p
  res_i = vec_any_eq(vuc, vuc);                 // CHECK: @llvm.ppc.altivec.vcmpequb.p
  res_i = vec_any_eq(vs, vs);                   // CHECK: @llvm.ppc.altivec.vcmpequh.p
  res_i = vec_any_eq(vus, vus);                 // CHECK: @llvm.ppc.altivec.vcmpequh.p
  res_i = vec_any_eq(vi, vi);                   // CHECK: @llvm.ppc.altivec.vcmpequw.p
  res_i = vec_any_eq(vui, vui);                 // CHECK: @llvm.ppc.altivec.vcmpequw.p
  res_i = vec_any_eq(vf, vf);                   // CHECK: @llvm.ppc.altivec.vcmpeqfp.p

  /* vec_any_ge */
  res_i = vec_any_ge(vsc, vsc);                 // CHECK: @llvm.ppc.altivec.vcmpgtsb.p
  res_i = vec_any_ge(vuc, vuc);                 // CHECK: @llvm.ppc.altivec.vcmpgtub.p
  res_i = vec_any_ge(vs, vs);                   // CHECK: @llvm.ppc.altivec.vcmpgtsh.p
  res_i = vec_any_ge(vus, vus);                 // CHECK: @llvm.ppc.altivec.vcmpgtuh.p
  res_i = vec_any_ge(vi, vi);                   // CHECK: @llvm.ppc.altivec.vcmpgtsw.p
  res_i = vec_any_ge(vui, vui);                 // CHECK: @llvm.ppc.altivec.vcmpgtuw.p
  res_i = vec_any_ge(vf, vf);                   // CHECK: @llvm.ppc.altivec.vcmpgtfp.p

  /* vec_any_gt */
  res_i = vec_any_gt(vsc, vsc);                 // CHECK: @llvm.ppc.altivec.vcmpgtsb.p
  res_i = vec_any_gt(vuc, vuc);                 // CHECK: @llvm.ppc.altivec.vcmpgtub.p
  res_i = vec_any_gt(vs, vs);                   // CHECK: @llvm.ppc.altivec.vcmpgtsh.p
  res_i = vec_any_gt(vus, vus);                 // CHECK: @llvm.ppc.altivec.vcmpgtuh.p
  res_i = vec_any_gt(vi, vi);                   // CHECK: @llvm.ppc.altivec.vcmpgtsw.p
  res_i = vec_any_gt(vui, vui);                 // CHECK: @llvm.ppc.altivec.vcmpgtuw.p
  res_i = vec_any_gt(vf, vf);                   // CHECK: @llvm.ppc.altivec.vcmpgtfp.p

  /* vec_any_le */
  res_i = vec_any_le(vsc, vsc);                 // CHECK: @llvm.ppc.altivec.vcmpgtsb.p
  res_i = vec_any_le(vuc, vuc);                 // CHECK: @llvm.ppc.altivec.vcmpgtub.p
  res_i = vec_any_le(vs, vs);                   // CHECK: @llvm.ppc.altivec.vcmpgtsh.p
  res_i = vec_any_le(vus, vus);                 // CHECK: @llvm.ppc.altivec.vcmpgtuh.p
  res_i = vec_any_le(vi, vi);                   // CHECK: @llvm.ppc.altivec.vcmpgtsw.p
  res_i = vec_any_le(vui, vui);                 // CHECK: @llvm.ppc.altivec.vcmpgtuw.p
  res_i = vec_any_le(vf, vf);                   // CHECK: @llvm.ppc.altivec.vcmpgtfp.p

  /* vec_any_lt */
  res_i = vec_any_lt(vsc, vsc);                 // CHECK: @llvm.ppc.altivec.vcmpgtsb.p
  res_i = vec_any_lt(vuc, vuc);                 // CHECK: @llvm.ppc.altivec.vcmpgtub.p
  res_i = vec_any_lt(vs, vs);                   // CHECK: @llvm.ppc.altivec.vcmpgtsh.p
  res_i = vec_any_lt(vus, vus);                 // CHECK: @llvm.ppc.altivec.vcmpgtuh.p
  res_i = vec_any_lt(vi, vi);                   // CHECK: @llvm.ppc.altivec.vcmpgtsw.p
  res_i = vec_any_lt(vui, vui);                 // CHECK: @llvm.ppc.altivec.vcmpgtuw.p
  res_i = vec_any_lt(vf, vf);                   // CHECK: @llvm.ppc.altivec.vcmpgtfp.p

  /* vec_any_nan */
  res_i = vec_any_nan(vf);                      // CHECK: @llvm.ppc.altivec.vcmpeqfp.p

  /* vec_any_ne */
  res_i = vec_any_ne(vsc, vsc);                 // CHECK: @llvm.ppc.altivec.vcmpequb.p
  res_i = vec_any_ne(vuc, vuc);                 // CHECK: @llvm.ppc.altivec.vcmpequb.p
  res_i = vec_any_ne(vs, vs);                   // CHECK: @llvm.ppc.altivec.vcmpequh.p
  res_i = vec_any_ne(vus, vus);                 // CHECK: @llvm.ppc.altivec.vcmpequh.p
  res_i = vec_any_ne(vi, vi);                   // CHECK: @llvm.ppc.altivec.vcmpequw.p
  res_i = vec_any_ne(vui, vui);                 // CHECK: @llvm.ppc.altivec.vcmpequw.p
  res_i = vec_any_ne(vf, vf);                   // CHECK: @llvm.ppc.altivec.vcmpeqfp.p

  /* vec_any_nge */
  res_i = vec_any_nge(vf, vf);                  // CHECK: @llvm.ppc.altivec.vcmpgefp.p

  /* vec_any_ngt */
  res_i = vec_any_ngt(vf, vf);                  // CHECK: @llvm.ppc.altivec.vcmpgtfp.p

  /* vec_any_nle */
  res_i = vec_any_nle(vf, vf);                  // CHECK: @llvm.ppc.altivec.vcmpgefp.p

  /* vec_any_nlt */
  res_i = vec_any_nlt(vf, vf);                  // CHECK: @llvm.ppc.altivec.vcmpgtfp.p

  /* vec_any_numeric */
  res_i = vec_any_numeric(vf);                  // CHECK: @llvm.ppc.altivec.vcmpeqfp.p

  /* vec_any_out */
  res_i = vec_any_out(vf, vf);                  // CHECK: @llvm.ppc.altivec.vcmpbfp.p

  return 0;
}
