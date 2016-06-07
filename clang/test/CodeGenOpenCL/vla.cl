// RUN: %clang_cc1 -emit-llvm -O0 -cl-std=CL2.0 -o - %s | FileCheck %s

constant int sz0 = 5;
// CHECK: @sz0 = constant i32 5, align 4
const global int sz1 = 16;
// CHECK: @sz1 = constant i32 16, align 4
const constant int sz2 = 8;
// CHECK: @sz2 = constant i32 8, align 4
// CHECK: @testvla.vla2 = internal global [8 x i16] undef, align 16

kernel void testvla()
{
  int vla0[sz0];
// CHECK: %vla0 = alloca [5 x i32], align 16
  char vla1[sz1];
// CHECK: %vla1 = alloca [16 x i8], align 16
  local short vla2[sz2];
}
