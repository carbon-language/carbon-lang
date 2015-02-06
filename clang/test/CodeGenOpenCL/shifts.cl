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

typedef __attribute__((ext_vector_type(4))) int int4;

//CHECK: @vectorVectorTest
int4 vectorVectorTest(int4 a,int4 b) {
  //CHECK: [[VM:%.+]] = and <4 x i32> %b, <i32 31, i32 31, i32 31, i32 31>
  //CHECK-NEXT: [[VC:%.+]] = shl <4 x i32> %a, [[VM]]
  int4 c = a << b;
  //CHECK-NEXT: [[VF:%.+]] = add <4 x i32> [[VC]], <i32 2, i32 4, i32 16, i32 8>
  int4 d = {1, 1, 1, 1};
  int4 e = {33, 34, -28, -29};
  int4 f = c + (d << e);
  //CHECK-NEXT: ret <4 x i32> [[VF]]
  return f;
}

//CHECK: @vectorScalarTest
int4 vectorScalarTest(int4 a,int b) {
  //CHECK: [[SP0:%.+]] = insertelement <4 x i32> undef, i32 %b, i32 0
  //CHECK: [[SP1:%.+]] = shufflevector <4 x i32> [[SP0]], <4 x i32> undef, <4 x i32> zeroinitializer
  //CHECK: [[VSM:%.+]] = and <4 x i32> [[SP1]], <i32 31, i32 31, i32 31, i32 31>
  //CHECK-NEXT: [[VSC:%.+]] = shl <4 x i32> %a, [[VSM]]
  int4 c = a << b;
  //CHECK-NEXT: [[VSF:%.+]] = add <4 x i32> [[VSC]], <i32 4, i32 4, i32 4, i32 4>
  int4 d = {1, 1, 1, 1};
  int4 f = c + (d << 34);
  //CHECK-NEXT: ret <4 x i32> [[VSF]]
  return f;
}
