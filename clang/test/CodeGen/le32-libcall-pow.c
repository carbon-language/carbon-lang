// RUN: %clang_cc1 -fmath-errno -emit-llvm -o - %s -triple le32-unknown-nacl | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -o - %s -triple le32-unknown-nacl | FileCheck %s

// le32 (PNaCl) never generates intrinsics for pow calls, with or without errno

// CHECK: define void @test_pow
void test_pow(float a0, double a1, long double a2) {
  // CHECK: call float @powf
  float l0 = powf(a0, a0);

  // CHECK: call double @pow
  double l1 = pow(a1, a1);

  // CHECK: call double @powl
  long double l2 = powl(a2, a2);
}

// CHECK: declare float @powf(float, float)
// CHECK: declare double @pow(double, double)
// CHECK: declare double @powl(double, double)

