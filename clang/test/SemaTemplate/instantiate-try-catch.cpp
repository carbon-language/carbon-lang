// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -std=c++0x -verify %s

template<typename T> struct TryCatch0 {
  void f() {
    try {
    } catch (T&&) { // expected-error 2{{cannot catch exceptions by rvalue reference}}
    }
  }
};

template struct TryCatch0<int&>; // okay
template struct TryCatch0<int&&>; // expected-note{{instantiation}}
template struct TryCatch0<int>; // expected-note{{instantiation}}

