// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -faltivec -target-feature +power8-vector -triple powerpc64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -faltivec -target-feature +power8-vector -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK-LE
// RUN: not %clang_cc1 -faltivec -triple powerpc64-unknown-unknown -emit-llvm %s -o - 2>&1 | FileCheck %s -check-prefix=CHECK-PPC

vector int vi = { -1, 2, -3, 4 };
vector unsigned int vui = { 1, 2, 3, 4 };
vector bool long long vbll = { 1, 0 };
vector long long vll = { 1, 2 };
vector unsigned long long vull = { 1, 2 };

int res_i;
vector int res_vi;
vector unsigned int res_vui;
vector long long res_vll;
vector unsigned long long res_vull;
vector bool long long res_vbll;

// CHECK-LABEL: define void @test1
void test1() {

  /* vec_cmpeq */
  res_vbll = vec_cmpeq(vll, vll);
// CHECK: @llvm.ppc.altivec.vcmpequd
// CHECK-LE: @llvm.ppc.altivec.vcmpequd
// CHECK-PPC: error: call to 'vec_cmpeq' is ambiguous

  res_vbll = vec_cmpeq(vull, vull);
// CHECK: @llvm.ppc.altivec.vcmpequd
// CHECK-LE: @llvm.ppc.altivec.vcmpequd
// CHECK-PPC: error: call to 'vec_cmpeq' is ambiguous

  /* vec_cmpgt */
  res_vbll = vec_cmpgt(vll, vll);
// CHECK: @llvm.ppc.altivec.vcmpgtsd
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsd
// CHECK-PPC: error: call to 'vec_cmpgt' is ambiguous

  res_vbll = vec_cmpgt(vull, vull);
// CHECK: @llvm.ppc.altivec.vcmpgtud
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud
// CHECK-PPC: error: call to 'vec_cmpgt' is ambiguous

  /* ----------------------- predicates --------------------------- */
  /* vec_all_eq */
  res_i = vec_all_eq(vll, vll);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_all_eq' is ambiguous

  res_i = vec_all_eq(vll, vbll);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_all_eq' is ambiguous

  res_i = vec_all_eq(vull, vull);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_all_eq' is ambiguous

  res_i = vec_all_eq(vull, vbll);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_all_eq' is ambiguous

  res_i = vec_all_eq(vbll, vll);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_all_eq' is ambiguous

  res_i = vec_all_eq(vbll, vull);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_all_eq' is ambiguous

  res_i = vec_all_eq(vbll, vbll);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_all_eq' is ambiguous

  /* vec_all_ne */
  res_i = vec_all_ne(vll, vll);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_all_ne' is ambiguous

  res_i = vec_all_ne(vll, vbll);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_all_ne' is ambiguous

  res_i = vec_all_ne(vull, vull);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_all_ne' is ambiguous

  res_i = vec_all_ne(vull, vbll);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_all_ne' is ambiguous

  res_i = vec_all_ne(vbll, vll);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_all_ne' is ambiguous

  res_i = vec_all_ne(vbll, vull);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_all_ne' is ambiguous

  res_i = vec_all_ne(vbll, vbll);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_all_ne' is ambiguous

  /* vec_any_eq */
  res_i = vec_any_eq(vll, vll);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_any_eq' is ambiguous

  res_i = vec_any_eq(vll, vbll);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_any_eq' is ambiguous

  res_i = vec_any_eq(vull, vull);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_any_eq' is ambiguous

  res_i = vec_any_eq(vull, vbll);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_any_eq' is ambiguous

  res_i = vec_any_eq(vbll, vll);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_any_eq' is ambiguous

  res_i = vec_any_eq(vbll, vull);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_any_eq' is ambiguous

  res_i = vec_any_eq(vbll, vbll);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_any_eq' is ambiguous

  /* vec_any_ne */
  res_i = vec_any_ne(vll, vll);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_any_ne' is ambiguous

  res_i = vec_any_ne(vll, vbll);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_any_ne' is ambiguous

  res_i = vec_any_ne(vull, vull);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_any_ne' is ambiguous

  res_i = vec_any_ne(vull, vbll);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_any_ne' is ambiguous

  res_i = vec_any_ne(vbll, vll);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_any_ne' is ambiguous

  res_i = vec_any_ne(vbll, vull);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_any_ne' is ambiguous

  res_i = vec_any_ne(vbll, vbll);
// CHECK: @llvm.ppc.altivec.vcmpequd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpequd.p
// CHECK-PPC: error: call to 'vec_any_ne' is ambiguous

  /* vec_all_ge */
  res_i = vec_all_ge(vll, vll);
// CHECK: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-PPC: error: call to 'vec_all_ge' is ambiguous

  res_i = vec_all_ge(vll, vbll);
// CHECK: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-PPC: error: call to 'vec_all_ge' is ambiguous

  res_i = vec_all_ge(vull, vull);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_all_ge' is ambiguous

  res_i = vec_all_ge(vull, vbll);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_all_ge' is ambiguous

  res_i = vec_all_ge(vbll, vll);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_all_ge' is ambiguous

  res_i = vec_all_ge(vbll, vull);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_all_ge' is ambiguous

  res_i = vec_all_ge(vbll, vbll);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_all_ge' is ambiguous

  /* vec_all_gt */
  res_i = vec_all_gt(vll, vll);
// CHECK: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-PPC: error: call to 'vec_all_gt' is ambiguous

  res_i = vec_all_gt(vll, vbll);
// CHECK: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-PPC: error: call to 'vec_all_gt' is ambiguous

  res_i = vec_all_gt(vull, vull);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_all_gt' is ambiguous

  res_i = vec_all_gt(vull, vbll);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_all_gt' is ambiguous

  res_i = vec_all_gt(vbll, vll);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_all_gt' is ambiguous

  res_i = vec_all_gt(vbll, vull);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_all_gt' is ambiguous

  res_i = vec_all_gt(vbll, vbll);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_all_gt' is ambiguous

  /* vec_all_le */
  res_i = vec_all_le(vll, vll);
// CHECK: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-PPC: error: call to 'vec_all_le' is ambiguous

  res_i = vec_all_le(vll, vbll);
// CHECK: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-PPC: error: call to 'vec_all_le' is ambiguous

  res_i = vec_all_le(vull, vull);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_all_le' is ambiguous

  res_i = vec_all_le(vull, vbll);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_all_le' is ambiguous

  res_i = vec_all_le(vbll, vll);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_all_le' is ambiguous

  res_i = vec_all_le(vbll, vull);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_all_le' is ambiguous

  res_i = vec_all_le(vbll, vbll);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_all_le' is ambiguous

  /* vec_all_lt */
  res_i = vec_all_lt(vll, vll);
// CHECK: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-PPC: error: call to 'vec_all_lt' is ambiguous

  res_i = vec_all_lt(vll, vbll);
// CHECK: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-PPC: error: call to 'vec_all_lt' is ambiguous

  res_i = vec_all_lt(vull, vull);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_all_lt' is ambiguous

  res_i = vec_all_lt(vull, vbll);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_all_lt' is ambiguous

  res_i = vec_all_lt(vbll, vll);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_all_lt' is ambiguous

  res_i = vec_all_lt(vbll, vull);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_all_lt' is ambiguous

  res_i = vec_all_lt(vbll, vbll);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_all_lt' is ambiguous

  /* vec_any_ge */
  res_i = vec_any_ge(vll, vll);
// CHECK: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-PPC: error: call to 'vec_any_ge' is ambiguous

  res_i = vec_any_ge(vll, vbll);
// CHECK: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-PPC: error: call to 'vec_any_ge' is ambiguous

  res_i = vec_any_ge(vull, vull);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_any_ge' is ambiguous

  res_i = vec_any_ge(vull, vbll);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_any_ge' is ambiguous

  res_i = vec_any_ge(vbll, vll);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_any_ge' is ambiguous

  res_i = vec_any_ge(vbll, vull);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_any_ge' is ambiguous

  res_i = vec_any_ge(vbll, vbll);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_any_ge' is ambiguous

  /* vec_any_gt */
  res_i = vec_any_gt(vll, vll);
// CHECK: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-PPC: error: call to 'vec_any_gt' is ambiguous

  res_i = vec_any_gt(vll, vbll);
// CHECK: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-PPC: error: call to 'vec_any_gt' is ambiguous

  res_i = vec_any_gt(vull, vull);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_any_gt' is ambiguous

  res_i = vec_any_gt(vull, vbll);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_any_gt' is ambiguous

  res_i = vec_any_gt(vbll, vll);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_any_gt' is ambiguous

  res_i = vec_any_gt(vbll, vull);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_any_gt' is ambiguous

  res_i = vec_any_gt(vbll, vbll);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_any_gt' is ambiguous

  /* vec_any_le */
  res_i = vec_any_le(vll, vll);
// CHECK: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-PPC: error: call to 'vec_any_le' is ambiguous

  res_i = vec_any_le(vll, vbll);
// CHECK: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-PPC: error: call to 'vec_any_le' is ambiguous

  res_i = vec_any_le(vull, vull);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_any_le' is ambiguous

  res_i = vec_any_le(vull, vbll);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_any_le' is ambiguous

  res_i = vec_any_le(vbll, vll);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_any_le' is ambiguous

  res_i = vec_any_le(vbll, vull);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_any_le' is ambiguous

  res_i = vec_any_le(vbll, vbll);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_any_le' is ambiguous

  /* vec_any_lt */
  res_i = vec_any_lt(vll, vll);
// CHECK: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-PPC: error: call to 'vec_any_lt' is ambiguous

  res_i = vec_any_lt(vll, vbll);
// CHECK: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtsd.p
// CHECK-PPC: error: call to 'vec_any_lt' is ambiguous

  res_i = vec_any_lt(vull, vull);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_any_lt' is ambiguous

  res_i = vec_any_lt(vull, vbll);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_any_lt' is ambiguous

  res_i = vec_any_lt(vbll, vll);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_any_lt' is ambiguous

  res_i = vec_any_lt(vbll, vull);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_any_lt' is ambiguous

  res_i = vec_any_lt(vbll, vbll);
// CHECK: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-LE: @llvm.ppc.altivec.vcmpgtud.p
// CHECK-PPC: error: call to 'vec_any_lt' is ambiguous

  /* vec_max */
  res_vll = vec_max(vll, vll);
// CHECK: @llvm.ppc.altivec.vmaxsd
// CHECK-LE: @llvm.ppc.altivec.vmaxsd
// CHECK-PPC: error: call to 'vec_max' is ambiguous

  res_vll = vec_max(vbll, vll);
// CHECK: @llvm.ppc.altivec.vmaxsd
// CHECK-LE: @llvm.ppc.altivec.vmaxsd
// CHECK-PPC: error: call to 'vec_max' is ambiguous

  res_vll = vec_max(vll, vbll);
// CHECK: @llvm.ppc.altivec.vmaxsd
// CHECK-LE: @llvm.ppc.altivec.vmaxsd
// CHECK-PPC: error: call to 'vec_max' is ambiguous

  res_vull = vec_max(vull, vull);
// CHECK: @llvm.ppc.altivec.vmaxud
// CHECK-LE: @llvm.ppc.altivec.vmaxud
// CHECK-PPC: error: call to 'vec_max' is ambiguous

  res_vull = vec_max(vbll, vull);
// CHECK: @llvm.ppc.altivec.vmaxud
// CHECK-LE: @llvm.ppc.altivec.vmaxud
// CHECK-PPC: error: call to 'vec_max' is ambiguous

  res_vull = vec_max(vull, vbll);
// CHECK: @llvm.ppc.altivec.vmaxud
// CHECK-LE: @llvm.ppc.altivec.vmaxud
// CHECK-PPC: error: call to 'vec_max' is ambiguous

  /* vec_min */
  res_vll = vec_min(vll, vll);
// CHECK: @llvm.ppc.altivec.vminsd
// CHECK-LE: @llvm.ppc.altivec.vminsd
// CHECK-PPC: error: call to 'vec_min' is ambiguous

  res_vll = vec_min(vbll, vll);
// CHECK: @llvm.ppc.altivec.vminsd
// CHECK-LE: @llvm.ppc.altivec.vminsd
// CHECK-PPC: error: call to 'vec_min' is ambiguous

  res_vll = vec_min(vll, vbll);
// CHECK: @llvm.ppc.altivec.vminsd
// CHECK-LE: @llvm.ppc.altivec.vminsd
// CHECK-PPC: error: call to 'vec_min' is ambiguous

  res_vull = vec_min(vull, vull);
// CHECK: @llvm.ppc.altivec.vminud
// CHECK-LE: @llvm.ppc.altivec.vminud
// CHECK-PPC: error: call to 'vec_min' is ambiguous

  res_vull = vec_min(vbll, vull);
// CHECK: @llvm.ppc.altivec.vminud
// CHECK-LE: @llvm.ppc.altivec.vminud
// CHECK-PPC: error: call to 'vec_min' is ambiguous

  res_vull = vec_min(vull, vbll);
// CHECK: @llvm.ppc.altivec.vminud
// CHECK-LE: @llvm.ppc.altivec.vminud
// CHECK-PPC: error: call to 'vec_min' is ambiguous

  /* vec_mule */
  res_vll = vec_mule(vi, vi);
// CHECK: @llvm.ppc.altivec.vmulesw
// CHECK-LE: @llvm.ppc.altivec.vmulosw
// CHECK-PPC: error: call to 'vec_mule' is ambiguous

  res_vull = vec_mule(vui , vui);
// CHECK: @llvm.ppc.altivec.vmuleuw
// CHECK-LE: @llvm.ppc.altivec.vmulouw
// CHECK-PPC: error: call to 'vec_mule' is ambiguous

  /* vec_mulo */
  res_vll = vec_mulo(vi, vi);
// CHECK: @llvm.ppc.altivec.vmulosw
// CHECK-LE: @llvm.ppc.altivec.vmulesw
// CHECK-PPC: error: call to 'vec_mulo' is ambiguous

  res_vull = vec_mulo(vui, vui);
// CHECK: @llvm.ppc.altivec.vmulouw
// CHECK-LE: @llvm.ppc.altivec.vmuleuw
// CHECK-PPC: error: call to 'vec_mulo' is ambiguous

  /* vec_rl */
  res_vll = vec_rl(vll, vull);
// CHECK: @llvm.ppc.altivec.vrld
// CHECK-LE: @llvm.ppc.altivec.vrld
// CHECK-PPC: error: call to 'vec_rl' is ambiguous

  res_vull = vec_rl(vull, vull);
// CHECK: @llvm.ppc.altivec.vrld
// CHECK-LE: @llvm.ppc.altivec.vrld
// CHECK-PPC: error: call to 'vec_rl' is ambiguous

  /* vec_sl */
  res_vll = vec_sl(vll, vull);
// CHECK: shl <2 x i64>
// CHECK-LE: shl <2 x i64>
// CHECK-PPC: error: call to 'vec_sl' is ambiguous

  res_vull = vec_sl(vull, vull);
// CHECK: shl <2 x i64>
// CHECK-LE: shl <2 x i64>
// CHECK-PPC: error: call to 'vec_sl' is ambiguous

  /* vec_sr */
  res_vll = vec_sr(vll, vull);
// CHECK: ashr <2 x i64>
// CHECK-LE: ashr <2 x i64>
// CHECK-PPC: error: call to 'vec_sr' is ambiguous

  res_vull = vec_sr(vull, vull);
// CHECK: lshr <2 x i64>
// CHECK-LE: lshr <2 x i64>
// CHECK-PPC: error: call to 'vec_sr' is ambiguous

  /* vec_sra */
  res_vll = vec_sra(vll, vull);
// CHECK: ashr <2 x i64>
// CHECK-LE: ashr <2 x i64>
// CHECK-PPC: error: call to 'vec_sra' is ambiguous

  res_vull = vec_sra(vull, vull);
// CHECK: ashr <2 x i64>
// CHECK-LE: ashr <2 x i64>
// CHECK-PPC: error: call to 'vec_sra' is ambiguous

}
