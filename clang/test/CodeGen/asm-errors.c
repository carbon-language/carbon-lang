// RUN: not %clang_cc1 -triple i386-apple-darwin10 -emit-obj %s  > %t 2>&1
// RUN: FileCheck %s < %t

int test1(int X) {
// CHECK: error: unrecognized instruction
  __asm__ ("abc incl    %0" : "+r" (X));
  return X;
}
