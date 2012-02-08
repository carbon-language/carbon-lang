// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -pedantic %s
float f = 0x1p+1; // expected-warning{{hexadecimal floating constants are a C99 feature}}
double e = 0x.p0; //expected-error{{hexadecimal floating constants require a significand}}
double d = 0x.2p2; // expected-warning{{hexadecimal floating constants are a C99 feature}}
float g = 0x1.2p2; // expected-warning{{hexadecimal floating constants are a C99 feature}}
double h = 0x1.p2; // expected-warning{{hexadecimal floating constants are a C99 feature}}
