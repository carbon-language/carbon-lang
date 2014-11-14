// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -faltivec -target-feature +vsx -triple powerpc64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

vector unsigned char vuc = { 8,  9, 10, 11, 12, 13, 14, 15,
                             0,  1,  2,  3,  4,  5,  6,  7};
vector float vf = { -1.5, 2.5, -3.5, 4.5 };
vector double vd = { 3.5, -7.5 };
vector signed int vsi = { -1, 2, -3, 4 };
vector unsigned int vui = { 0, 1, 2, 3 };
vector signed long long vsll = { 255LL, -937LL };
vector unsigned long long vull = { 1447LL, 2894LL };
double d = 23.4;

vector float res_vf;
vector double res_vd;
vector signed int res_vsi;
vector unsigned int res_vui;
vector signed long long res_vsll;
vector unsigned long long res_vull;
double res_d;

void test1() {
// CHECK-LABEL: define void @test1

  /* vec_div */
  res_vf = vec_div(vf, vf);
// CHECK: @llvm.ppc.vsx.xvdivsp

  res_vd = vec_div(vd, vd);
// CHECK: @llvm.ppc.vsx.xvdivdp

  /* vec_max */
  res_vf = vec_max(vf, vf);
// CHECK: @llvm.ppc.vsx.xvmaxsp

  res_vd = vec_max(vd, vd);
// CHECK: @llvm.ppc.vsx.xvmaxdp

  res_vf = vec_vmaxfp(vf, vf);
// CHECK: @llvm.ppc.vsx.xvmaxsp

  /* vec_min */
  res_vf = vec_min(vf, vf);
// CHECK: @llvm.ppc.vsx.xvminsp

  res_vd = vec_min(vd, vd);
// CHECK: @llvm.ppc.vsx.xvmindp

  res_vf = vec_vminfp(vf, vf);
// CHECK: @llvm.ppc.vsx.xvminsp

  res_d = __builtin_vsx_xsmaxdp(d, d);
// CHECK: @llvm.ppc.vsx.xsmaxdp

  res_d = __builtin_vsx_xsmindp(d, d);
// CHECK: @llvm.ppc.vsx.xsmindp

  /* vec_perm */
  res_vsll = vec_perm(vsll, vsll, vuc);
// CHECK: @llvm.ppc.altivec.vperm

  res_vull = vec_perm(vull, vull, vuc);
// CHECK: @llvm.ppc.altivec.vperm

  res_vd = vec_perm(vd, vd, vuc);
// CHECK: @llvm.ppc.altivec.vperm

  res_vsll = vec_vperm(vsll, vsll, vuc);
// CHECK: @llvm.ppc.altivec.vperm

  res_vull = vec_vperm(vull, vull, vuc);
// CHECK: @llvm.ppc.altivec.vperm

  res_vd = vec_vperm(vd, vd, vuc);
// CHECK: @llvm.ppc.altivec.vperm

  /* vec_vsx_ld */

  res_vsi = vec_vsx_ld(0, &vsi);
// CHECK: @llvm.ppc.vsx.lxvw4x

  res_vui = vec_vsx_ld(0, &vui);
// CHECK: @llvm.ppc.vsx.lxvw4x

  res_vf = vec_vsx_ld (0, &vf);
// CHECK: @llvm.ppc.vsx.lxvw4x

  res_vsll = vec_vsx_ld(0, &vsll);
// CHECK: @llvm.ppc.vsx.lxvd2x

  res_vull = vec_vsx_ld(0, &vull);
// CHECK: @llvm.ppc.vsx.lxvd2x

  res_vd = vec_vsx_ld(0, &vd);
// CHECK: @llvm.ppc.vsx.lxvd2x

  /* vec_vsx_st */

  vec_vsx_st(vsi, 0, &res_vsi);
// CHECK: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vui, 0, &res_vui);
// CHECK: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vf, 0, &res_vf);
// CHECK: @llvm.ppc.vsx.stxvw4x

  vec_vsx_st(vsll, 0, &res_vsll);
// CHECK: @llvm.ppc.vsx.stxvd2x

  vec_vsx_st(vull, 0, &res_vull);
// CHECK: @llvm.ppc.vsx.stxvd2x

  vec_vsx_st(vd, 0, &res_vd);
// CHECK: @llvm.ppc.vsx.stxvd2x
}
