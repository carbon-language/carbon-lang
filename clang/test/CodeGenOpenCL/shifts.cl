// RUN: %clang_cc1 -x cl -O1 -emit-llvm  %s -o - -triple x86_64-linux-gnu | FileCheck %s
// OpenCL essentially reduces all shift amounts to the last word-size bits before evaluating.
// Test this both for variables and constants evaluated in the front-end.


//CHECK: @positiveShift32
int positiveShift32(int a,int b) {
  //CHECK: %shl.mask = and i32 %b, 31
  //CHECK-NEXT: %shl = shl i32 %a, %shl.mask
  int c = a<<b;
  int d = ((int)1)<<33;
  //CHECK-NEXT: %add = add nsw i32 %shl, 2
  int e = c + d;
  //CHECK-NEXT: ret i32 %add
  return e;
}

//CHECK: @positiveShift64
long positiveShift64(long a,long b) {
  //CHECK: %shr.mask = and i64 %b, 63
  //CHECK-NEXT: %shr = ashr i64 %a, %shr.mask
  long c = a>>b;
  long d = ((long)8)>>65;
  //CHECK-NEXT: %add = add nsw i64 %shr, 4
  long e = c + d;
  //CHECK-NEXT: ret i64 %add
  return e;
}
