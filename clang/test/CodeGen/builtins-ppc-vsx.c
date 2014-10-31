// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -faltivec -target-feature +vsx -triple powerpc64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

vector float vf = { -1.5, 2.5, -3.5, 4.5 };
vector double vd = { 3.5, -7.5 };
double d = 23.4;

vector float res_vf;
vector double res_vd;
double res_d;

void test1() {
// CHECK-LABEL: define void @test1

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
}
