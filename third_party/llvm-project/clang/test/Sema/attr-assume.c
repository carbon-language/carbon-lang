// RUN: %clang_cc1 -triple i386-apple-darwin9 -fsyntax-only -verify %s

void f1() __attribute__((assume(3))); // expected-error {{'assume' attribute requires a string}}
void f2() __attribute__((assume(int))); // expected-error {{expected expression}}
void f3() __attribute__((assume(for))); // expected-error {{expected expression}}
void f4() __attribute__((assume("QQQQ"))); // expected-warning {{unknown assumption string 'QQQQ'; attribute is potentially ignored}}
void f5() __attribute__((assume("omp_no_openmp")));
void f6() __attribute__((assume("omp_noopenmp"))); // expected-warning {{unknown assumption string 'omp_noopenmp' may be misspelled; attribute is potentially ignored, did you mean 'omp_no_openmp'?}}
void f7() __attribute__((assume("omp_no_openmp_routine"))); // expected-warning {{unknown assumption string 'omp_no_openmp_routine' may be misspelled; attribute is potentially ignored, did you mean 'omp_no_openmp_routines'?}}
void f8() __attribute__((assume("omp_no_openmp1"))); // expected-warning {{unknown assumption string 'omp_no_openmp1' may be misspelled; attribute is potentially ignored, did you mean 'omp_no_openmp'?}}
void f9() __attribute__((assume("omp_no_openmp", "omp_no_openmp"))); // expected-error {{'assume' attribute takes one argument}}

int g1 __attribute__((assume(0))); // expected-warning {{'assume' attribute only applies to functions and Objective-C methods}}
int g2 __attribute__((assume("omp_no_openmp"))); // expected-warning {{'assume' attribute only applies to functions and Objective-C methods}}
