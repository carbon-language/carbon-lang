// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -pedantic %s
float f = 0x1p+1; // expected-warning{{hexadecimal floating constants are a C99 feature}}

