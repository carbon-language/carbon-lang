// RN: clang-cc -fsyntax-only -verify %s
// RUN: false
// XFAIL
template<int N> void f(int (&array)[N]);

template<> void f<1>(int (&array)[1]) { }
