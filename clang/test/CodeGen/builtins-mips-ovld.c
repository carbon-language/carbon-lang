// REQUIRES: mips-registered-target
// RUN: %clang_cc1 -triple mips-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

typedef signed char v4i8 __attribute__ ((vector_size(4)));

void foo() {
  v4i8 a = {1, 2, 3, 4};
  int shift = 1;
// CHECK: {{%.*}} = call <4 x i8> @llvm.mips.shll.qb(<4 x i8> {{%.*}}, i32 1)
  v4i8 r1 = __builtin_mips_shll_qb(a, 1);
// CHECK: {{%.*}} = call <4 x i8> @llvm.mips.shll.qb.v(<4 x i8> {{%.*}}, i32 {{%.*}})
  v4i8 r2 = __builtin_mips_shll_qb(a, shift);
}
