// RUN: clang-cc -fsyntax-only -fixit-at=fixit-at.c:3:1 %s -o - | clang-cc -verify -x c -

_Complex cd;

int i0 = { 17 }; // expected-warning{{braces}}
