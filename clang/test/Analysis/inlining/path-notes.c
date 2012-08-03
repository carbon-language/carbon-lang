// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-ipa=inlining -analyzer-output=text -verify %s

void zero(int **p) {
  *p = 0;
  // expected-note@-1 {{Null pointer value stored to 'a'}}
}

void testZero(int *a) {
  zero(&a);
  // expected-note@-1 {{Calling 'zero'}}
  // expected-note@-2 {{Returning from 'zero'}}
  *a = 1; // expected-warning{{Dereference of null pointer}}
  // expected-note@-1 {{Dereference of null pointer (loaded from variable 'a')}}
}