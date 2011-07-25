// RUN: %clang_cc1 -triple arm-unknown-unknown -target-abi apcs-gnu -fsyntax-only -verify %s

struct s0 { double f0; int f1; };
char chk0[__alignof__(struct s0) == 4 ? 1 : -1]; 

double g1;
short chk1[__alignof__(g1) == 4 ? 1 : -1]; 
short chk2[__alignof__(double) == 4 ? 1 : -1];

long long g2;
short chk1[__alignof__(g2) == 4 ? 1 : -1]; 
short chk2[__alignof__(long long) == 4 ? 1 : -1];

_Complex double g3;
short chk1[__alignof__(g3) == 4 ? 1 : -1]; 
short chk2[__alignof__(_Complex double) == 4 ? 1 : -1];
