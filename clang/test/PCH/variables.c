// Test this without pch.
// RUN: clang-cc -triple=i686-apple-darwin9 -include %S/variables.h -fsyntax-only -verify %s &&

// Test with pch.
// RUN: clang-cc -emit-pch -triple=i686-apple-darwin9 -o %t %S/variables.h &&
// RUN: clang-cc -triple=i686-apple-darwin9 -include-pch %t -fsyntax-only -verify %s 

int *ip2 = &x;
float *fp = &ip; // expected-warning{{incompatible pointer types}}
// FIXME:variables.h expected-note{{previous}}
double z; // expected-error{{redefinition}}

//double VeryHappy; // FIXME: xpected-error{{redefinition}}


int Q = A_MACRO_IN_THE_PCH;

int R = FUNCLIKE_MACRO(A_MACRO_, IN_THE_PCH);


int UNIQUE(a);  // a2
int *Arr[] = { &a0, &a1, &a2 };
