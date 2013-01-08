// RUN: %clang_cc1 -x cl -O1 -emit-llvm  %s -o - -triple x86_64-linux-gnu | FileCheck %s
// OpenCL essentially reduces all shift amounts to the last word-size bits before evaluating.
// Test this both for variables and constants evaluated in the front-end.


//CHECK: @positiveShift32
int positiveShift32(int a,int b) {
  //CHECK: [[M32:%.+]] = and i32 %b, 31
  //CHECK-NEXT: [[C32:%.+]] = shl i32 %a, [[M32]]
  int c = a<<b;
  int d = ((int)1)<<33;
  //CHECK-NEXT: [[E32:%.+]] = add nsw i32 [[C32]], 2
  int e = c + d;
  //CHECK-NEXT: ret i32 [[E32]]
  return e;
}

//CHECK: @positiveShift64
long positiveShift64(long a,long b) {
  //CHECK: [[M64:%.+]] = and i64 %b, 63
  //CHECK-NEXT: [[C64:%.+]] = ashr i64 %a, [[M64]]
  long c = a>>b;
  long d = ((long)8)>>65;
  //CHECK-NEXT: [[E64:%.+]] = add nsw i64 [[C64]], 4
  long e = c + d;
  //CHECK-NEXT: ret i64 [[E64]]
  return e;
}
