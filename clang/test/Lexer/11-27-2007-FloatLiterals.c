// RUN: clang-cc %s -emit-llvm -o - | grep 0x3BFD83C940000000 | count 2 &&
// RUN: clang-cc %s -emit-llvm -o - | grep 2.000000e+32 | count 2

float  F  = 1e-19f;
double D  = 2e32;
float  F2 = 01e-19f;
double D2 = 02e32;
