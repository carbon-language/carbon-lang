// RUN: %clang_cc1 -fixit-at=fixit-at.c:3:1 %s -o - | %clang_cc1 -verify -x c -

_Complex cd;

int i0[1] = { { 17 } }; // expected-warning{{braces}}
