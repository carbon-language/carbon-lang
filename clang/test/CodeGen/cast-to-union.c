// RUN: clang-cc -emit-llvm < %s -o %t &&
// RUN: grep "store i32 351, i32*" %t &&
// RUN: grep "w = global %0 { i32 2, \[4 x i8\] zeroinitializer }" %t &&
// RUN: grep "y = global %union.u { double 7.300000e+01 }" %t

union u { int i; double d; };

void foo() {
  union u ola = (union u) 351;
  union u olb = (union u) 1.0;
}

union u w = (union u)2;
union u y = (union u)73.0;
