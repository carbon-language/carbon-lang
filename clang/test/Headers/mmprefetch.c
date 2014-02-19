// RUN: %clang_cc1 -triple=x86_64-unknown-unknown -fms-extensions -emit-llvm -verify %s

#include <mmintrin.h>

// Check to make sure that _mm_prefetch survives redeclaration.
void _mm_prefetch(char const*, int);

void f(char *a) {
  _mm_prefetch(a, 0);
  _mm_prefetch(a, 1);
  _mm_prefetch(a, 2);
  _mm_prefetch(a, 3);
  _mm_prefetch(a, 4); // expected-error {{argument should be a value from 0 to 3}}
  _mm_prefetch(a, 0, 0); // expected-error {{too many arguments to function call, expected 2, have 3}}
  _mm_prefetch(a); // expected-error {{too few arguments to function call, expected 2, have 1}}
};
