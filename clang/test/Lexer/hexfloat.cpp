//RUN: %clang_cc1 -fsyntax-only -verify
//RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify

#ifndef __GXX_EXPERIMENTAL_CXX0X__
float f = 0x1p+1; // expected-warning {{incompatible with C++0x}}
#else
float f = 0x1p+1; // expected-warning {{invalid suffix}}
#endif
