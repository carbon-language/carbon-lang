// RUN: clang -emit-llvm < %s -o %t &&
// RUN: grep "store i32 351, i32*" %t &&
// RUN: grep "w = global <{ i32, i8, i8, i8, i8 }> <{ i32 2, i8 0, i8 0, i8 0, i8 0 }>" %t &&
// RUN: grep "y = global <{ double }> <{ double 7.300000e+01 }>" %t

union u { int i; double d; };

void foo() {
  union u ola = (union u) 351;
}

union u w = (union u)2;
union u y = (union u)73.0;
