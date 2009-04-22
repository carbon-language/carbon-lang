// Test this without pch.
// RUN: clang-cc -include %S/variables.h -fsyntax-only -verify %s &&

// Test with pch.
// RUN: clang-cc -emit-pch -o %t %S/variables.h &&
// RUN: clang-cc -include-pch %t -fsyntax-only -verify %s 

int *ip2 = &x;
float *fp = &ip; // expected-warning{{incompatible pointer types}}
// FIXME:variables.h expected-note{{previous}}
double z; // expected-error{{redefinition}}
// FIXME:variables.h expected-note{{previous}}
int z2 = 18; // expected-error{{redefinition}}
double VeryHappy; // expected-error{{redefinition}}
// FIXME:variables.h expected-note{{previous definition is here}}

int Q = A_MACRO_IN_THE_PCH;

int R = FUNCLIKE_MACRO(A_MACRO_, IN_THE_PCH);


int UNIQUE(a);  // a2
int *Arr[] = { &a0, &a1, &a2 };
