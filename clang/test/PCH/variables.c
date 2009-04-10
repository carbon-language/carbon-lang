// RUN: clang-cc -emit-pch -o %t %S/variables.h &&
// RUN: clang-cc -include-pch %t -fsyntax-only -verify %s 

int *ip2 = &x;
float *fp = &ip; // expected-warning{{incompatible pointer types}}
// FIXME:variables.h expected-note{{previous}}
double z; // expected-error{{redefinition}}

//double VeryHappy; // FIXME: xpected-error{{redefinition}}

