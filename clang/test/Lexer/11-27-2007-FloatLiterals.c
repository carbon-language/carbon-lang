// RUN: clang-cc %s -emit-llvm -o - | FileCheck %s

// CHECK: 0x3BFD83C940000000
// CHECK: 2.000000e+{{[0]*}}32
// CHECK: 0x3BFD83C940000000
// CHECK: 2.000000e+{{[0]*}}32

float  F  = 1e-19f;
double D  = 2e32;
float  F2 = 01e-19f;
double D2 = 02e32;
