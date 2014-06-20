// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

template<typename> struct S {};
template<typename> void f();


void foo(void) {
  // In C++11 mode, all of these are expected to parse correctly, and the CUDA
  // language should not interfere with that.

  // expected-no-diagnostics

  S<S<S<int>>> s3;

  S<S<S<S<int>>>> s4;

  S<S<S<S<S<int>>>>> s5;

  (void)(&f<S<S<int>>>==0);
}
