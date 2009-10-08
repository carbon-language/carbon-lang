// RUN: clang-cc -emit-llvm  %s -o - | FileCheck %s
// CHECK: w = global %0 { i32 2, [4 x i8] zeroinitializer }
// CHECK: y = global %union.u { double 7.300000e+0{{[0]*}}1 }
// CHECK: store i32 351, i32

union u { int i; double d; };

void foo() {
  union u ola = (union u) 351;
  union u olb = (union u) 1.0;
}

union u w = (union u)2;
union u y = (union u)73.0;
