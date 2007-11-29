// RUN: clang %s -emit-llvm 2>&1 | grep 0x3BFD83C940000000 | count 2
// RUN: clang %s -emit-llvm 2>&1 | grep 0x46A3B8B5B5056E16 | count 2

float  F  = 1e-19f;
double D  = 2e32;
float  F2 = 01e-19f;
double D2 = 02e32;
